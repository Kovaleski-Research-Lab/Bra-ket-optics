import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
from scipy.spatial.distance import cdist

def euclidean_distance(s, d) -> float:
    """
    Compute Euclidean distance between source and destination points.

    Args:
        s (array-like): Source point.
        d (array-like): Destination point.

    Returns:
        float: Euclidean distance.
    """
    s = np.asarray(s)
    d = np.asarray(d)
    return np.linalg.norm(d - s)

def direct_transfer_function(s, r, k, **kwargs) -> np.array:
    """
    Compute the transfer function G(s, r) using a nested-loop (direct) method.

    Args:
        s (np.ndarray): Source points (Ns, 3).
        r (np.ndarray): Receiver points (Nr, 3).
        k (float): Wavenumber.

    Returns:
        np.ndarray: Transfer matrix of shape (Nr, Ns).
    """
    g = np.zeros((len(r), len(s)), dtype=complex)
    for i, r_point in tqdm(enumerate(r), desc="Computing transfer function (direct method)", total=len(r)):
        for j, s_point in enumerate(s):
            d = euclidean_distance(s_point, r_point)
            numerator = -np.exp(1j * k * d)
            denominator = 4 * np.pi * d
            g[i, j] = np.round(numerator / denominator, 5)
    return g

def blockwise_transfer_function(s, r, k, blocksize=500, **kwargs) -> np.array:
    """
    Compute the transfer function using a blockwise vectorized method.

    Args:
        s (np.ndarray): Source points (Ns, 3).
        r (np.ndarray): Receiver points (Nr, 3).
        k (float): Wavenumber.
        blocksize (int): Number of receivers per block.

    Returns:
        np.ndarray: Transfer matrix of shape (Nr, Ns).
    """
    Ns, Nr = len(s), len(r)
    g = np.zeros((Nr, Ns), dtype=complex)
    for i in tqdm(range(0, Nr, blocksize), desc="Computing transfer function (blockwise method)", total=(Nr + blocksize - 1) // blocksize):
        r_r_block = r[i:i + blocksize]
        diffs = r_r_block[:, None, :] - s[None, :, :]  # Compute all vector diffs
        dists = np.linalg.norm(diffs, axis=-1)
        dists = np.where(dists == 0, np.finfo(float).eps, dists)  # Avoid division by zero
        g_block = -np.exp(1j * k * dists) / (4 * np.pi * dists)
        g[i:i + blocksize, :] = g_block
    return g

def sum_rule(g: np.array, *args, **kwargs) -> float:
    """
    Compute the sum rule (Frobenius norm squared) of the transfer matrix.

    Args:
        g (np.ndarray): Transfer matrix.

    Returns:
        float: Frobenius norm squared of g.
    """
    return np.linalg.norm(g, 'fro') ** 2

def free_space_transfer_function(s, r, k, method='direct', **kwargs) -> np.array:
    """
    Wrapper to compute the free-space transfer function using selected method.

    Args:
        s (np.ndarray): Source points (Ns, 3).
        r (np.ndarray): Receiver points (Nr, 3).
        k (float): Wavenumber.
        method (str): 'direct' or 'blockwise'.

    Returns:
        np.ndarray: Transfer matrix of shape (Nr, Ns).
    """
    if method == 'direct':
        return direct_transfer_function(s, r, k)
    elif method == 'blockwise':
        blocksize = kwargs.get('blocksize', 500)
        return blockwise_transfer_function(s, r, k, blocksize, **kwargs)
    else:
        raise ValueError("Method not recognized.")

def normalize_phase(eigenvector):
    """
    Normalize phase of an eigenvector such that its center value is real.

    Args:
        eigenvector (np.ndarray): Complex eigenvector.

    Returns:
        np.ndarray: Phase-normalized eigenvector.
    """
    mid_idx = len(eigenvector)//2
    phase = np.angle(eigenvector[mid_idx])
    return eigenvector * np.exp(-1j * phase)

def normalize_eigenvector(eigenvector):
    """
    Normalize all eigenvectors by aligning their phase.

    Args:
        eigenvector (np.ndarray): Array of eigenvectors (Ns, Nmodes).

    Returns:
        np.ndarray: Phase-normalized eigenvectors.
    """
    return np.column_stack([
        normalize_phase(eigenvector[:, j])
        for j in range(eigenvector.shape[1])
    ])

def evaluate_modes_direct(eigenvector, source_points, receiving_points, k, z_normalize=True):
    """
    Evaluate modes at receiver points using nested loops.

    Args:
        eigenvector (np.ndarray): Eigenvectors (Ns, Nmodes).
        source_points (np.ndarray): Source positions (Ns, 3).
        receiving_points (np.ndarray): Evaluation points (Nr, 3).
        k (float): Wavenumber.
        z_normalize (bool): Normalize by sqrt(z) if True.

    Returns:
        np.ndarray: Mode evaluations (Nmodes, Nr).
    """
    num_modes = eigenvector.shape[1]
    values = []

    for m in tqdm(range(num_modes), desc="Evaluating modes", total=num_modes):
        temp = []
        for p in tqdm(receiving_points, desc=f"Evaluating mode {m}", leave=False):
            h = eigenvector[:, m]
            val = 0
            for i, source in enumerate(source_points):
                distance = euclidean_distance(source, p)
                val += np.exp(1j*k*distance) * h[i] / distance
            val *= (-1 / (4 * np.pi))
            val *= np.sqrt(p[2]) if z_normalize else 1
            temp.append(val)
        values.append(temp)
    return np.asarray(values)

def evaluate_modes_vectorized(eigenvector, source_points, receiving_points, k, z_normalize=True):
    """
    Evaluate modes using a vectorized implementation.

    Args:
        eigenvector (np.ndarray): Eigenvectors (Ns, Nmodes).
        source_points (np.ndarray): Source points (Ns, 3).
        receiving_points (np.ndarray): Evaluation grid points (Nr, 3).
        k (float): Wavenumber.
        z_normalize (bool): Normalize by sqrt(z) if True.

    Returns:
        np.ndarray: Evaluated modes (Nmodes, Nr).
    """
    distances = cdist(receiving_points, source_points)  # (Nr, Ns)
    greens_kernel = np.exp(1j * k * distances) / distances  # (Nr, Ns)
    projected = greens_kernel @ eigenvector  # (Nr, Nmodes)
    scale = (-1 / (4 * np.pi)) * np.sqrt(np.abs(receiving_points[:, 2])) if z_normalize else (-1 / (4 * np.pi))
    return (scale * projected.T)  # (Nmodes, Nr)

def evaluate_modes(eigenvector, source_points, receiving_points, k, z_normalize=True, method='direct'):
    """
    Evaluate optical modes using the specified evaluation method.

    Args:
        eigenvector (np.ndarray): Eigenvectors.
        source_points (np.ndarray): Source points.
        receiving_points (np.ndarray): Receiver grid points.
        k (float): Wavenumber.
        method (str): 'direct' or 'vectorized'.

    Returns:
        np.ndarray: Mode evaluations.
    """
    if method == 'direct':
        return evaluate_modes_direct(eigenvector, source_points, receiving_points, k, z_normalize)
    elif method == 'vectorized':
        return evaluate_modes_vectorized(eigenvector, source_points, receiving_points, k, z_normalize)
    else:
        raise ValueError("Method not recognized. Use 'direct' or 'vectorized'.")

def plot_modes(xx, yy, zz, modes, plane='yz', x=None, y=None, z=None):
    """
    Plot the real part of evaluated optical modes in a specific 2D slice.

    Args:
        xx, yy, zz (np.ndarray): 3D meshgrid coordinate arrays.
        modes (np.ndarray): Evaluated mode values (Nmodes, ...).
        plane (str): 'yz', 'xz', or 'xy' for slice direction.
        x, y, z (int): Slice index along the fixed axis.
    """
    modes = np.asarray([np.reshape(i, zz.shape) for i in modes])

    if plane == 'yz':
        if x is None:
            raise ValueError("For 'yz' plane, x-coordinate must be specified.")
        for mode in modes:
            sliced_modes = mode[x, :, :].squeeze()
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(np.real(sliced_modes), cmap='jet')
            plt.show()

    elif plane == 'xz':
        sliced_modes = modes[:, :, 0, :].squeeze()
        # Optional: Add plot for xz

    elif plane == 'xy':
        sliced_modes = modes[:, :, :, 0].squeeze()
        # Optional: Add plot for xy

    else:
        raise ValueError("Plane must be 'yz', 'xz', or 'xy'.")

def matprint(mat, fmt="g"):
    """
    Nicely formatted print of a 2D matrix with column alignment.

    Args:
        mat (np.ndarray): Matrix to print.
        fmt (str): Format specifier.
    """
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")

# ========================== MAIN =============================== #

if __name__ == "__main__":
    # Define simulation parameters
    wavelength = 1
    Ns = 9
    dy = wavelength / 2
    Lz = 5 * wavelength
    Nr = 9
    k = 2 * np.pi / wavelength

    # Generate source and destination points
    s = np.asarray([(0, i * dy, 0) for i in range(Ns)])
    d = np.asarray([(0, i * dy, Lz) for i in range(Nr)])

    # Compute transfer function using blockwise method
    Gsr = free_space_transfer_function(s, d, k, method='blockwise')

    # Compare to theoretical scale factor
    gsr = np.round(-4 * np.pi * Lz, 2)
    S = sum_rule(Gsr)
    print(S)
    print(np.allclose(S, 72.65 / (gsr**2), atol=1e-5))

    # Eigen-decomposition of Gsr^H Gsr
    Gsrd_Gsr = np.matmul(np.matrix.getH(Gsr), Gsr)
    eig_vals, eig_vect = scipy.linalg.eigh(Gsrd_Gsr)
    eig_vals = np.flip(eig_vals)
    eig_vect = np.flip(eig_vect, axis=-1)
    eig_vect_normalized = eig_vect  # Use unnormalized for now

    # Create meshgrid to evaluate modes
    num_points = 200
    y = np.linspace(-dy, dy * Ns, num_points)
    z = np.linspace(0.2, Lz + 0.2, num_points)
    x = 0
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    # Evaluate the optical modes at the meshgrid points
    evaluated_modes = evaluate_modes(eig_vect_normalized, s, points, k, method='vectorized')

    # Plot the modes in the yz-plane at x=0
    plot_modes(xx, yy, zz, evaluated_modes, plane='yz', x=0)

