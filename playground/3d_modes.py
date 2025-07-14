import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
from scipy.spatial.distance import cdist

def euclidean_distance(s, d) -> float:
    """
    Computes the Euclidean distance between two points.

    Args:
        s (array-like): Source point.
        d (array-like): Destination point.

    Returns:
        float: Euclidean distance between s and d.
    """
    s = np.asarray(s)
    d = np.asarray(d)
    return np.linalg.norm(d - s)

def direct_transfer_function(s, r, k, **kwargs) -> np.array:
    """
    Computes the transfer function between sources and receivers using nested loops (direct method).

    Args:
        s (np.ndarray): Source points of shape (Ns, 3).
        r (np.ndarray): Receiving points of shape (Nr, 3).
        k (float): Wavenumber.

    Returns:
        np.ndarray: Transfer function matrix of shape (Nr, Ns).
    """
    g = np.zeros((len(r), len(s)), dtype=complex)
    for i, r_point in tqdm(enumerate(r), desc="Computing transfer function (direct method)", total=len(r)):
        for j, s_point in enumerate(s):
            d = euclidean_distance(r_point, s_point)
            numerator = -np.exp(1j * k * d)
            denominator = 4 * np.pi * d
            g[i,j] = numerator / denominator
    return g

def blockwise_transfer_function(s, r, k, blocksize=500, **kwargs) -> np.array:
    """
    Computes the transfer function in blocks for memory efficiency (vectorized within blocks).

    Args:
        s (np.ndarray): Source points (Ns, 3).
        r (np.ndarray): Receiving points (Nr, 3).
        k (float): Wavenumber.
        blocksize (int): Number of receiver points per block.

    Returns:
        np.ndarray: Transfer function matrix (Nr, Ns).
    """
    Ns, Nr = len(s), len(r)
    g = np.zeros((Nr, Ns), dtype=complex)
    for i in tqdm(range(0, Nr, blocksize), desc="Computing transfer function (blockwise method)", total=(Nr + blocksize - 1) // blocksize):
        r_r_block = r[i:i + blocksize]
        diffs = r_r_block[:, None, :] - s[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        dists = np.where(dists == 0, np.finfo(float).eps, dists)  # Avoid division by zero
        g_block = -np.exp(1j * k * dists) / (4 * np.pi * dists)
        g[i:i + blocksize, :] = g_block
    return g

def sum_rule(g: np.array, *args, **kwargs) -> float:
    """
    Computes the Frobenius norm squared (sum of squared magnitudes) of the transfer matrix.

    Args:
        g (np.ndarray): Transfer matrix.

    Returns:
        float: Frobenius norm squared.
    """
    return np.linalg.norm(g, 'fro') ** 2

def free_space_transfer_function(s, r, k, method='direct', **kwargs) -> np.array:
    """
    Computes the transfer function between source and receiving points using a specified method.

    Args:
        s (np.ndarray): Source points.
        r (np.ndarray): Receiving points.
        k (float): Wavenumber.
        method (str): 'direct' or 'blockwise'.

    Returns:
        np.ndarray: Transfer function matrix.
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
    Normalizes the phase of an eigenvector so that its center element is real-valued.

    Args:
        eigenvector (np.ndarray): 1D complex eigenvector.

    Returns:
        np.ndarray: Phase-normalized eigenvector.
    """
    mid_idx = len(eigenvector) // 2
    phase = np.angle(eigenvector[mid_idx])
    return eigenvector * np.exp(-1j * phase)

def normalize_eigenvector(eigenvector):
    """
    Normalizes each column of an eigenvector matrix by phase.

    Args:
        eigenvector (np.ndarray): 2D array of eigenvectors.

    Returns:
        np.ndarray: Column-wise phase-normalized eigenvectors.
    """
    return np.column_stack([
        normalize_phase(eigenvector[:, j])
        for j in range(eigenvector.shape[1])
    ])

def evaluate_modes_direct(eigenvector, source_points, receiving_points, k, z_normalize=True):
    """
    Evaluate modes at receiving points using a nested-loop (non-vectorized) method.

    Args:
        eigenvector (np.ndarray): Eigenvectors (Ns, Nmodes).
        source_points (np.ndarray): Source point array (Ns, 3).
        receiving_points (np.ndarray): Evaluation points (Nr, 3).
        k (float): Wavenumber.
        z_normalize (bool): Whether to apply sqrt(z) normalization.

    Returns:
        np.ndarray: Mode evaluations of shape (Nmodes, Nr).
    """
    Ns = len(source_points)
    Nr = len(receiving_points)
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
    Evaluate modes at receiving points using a fully vectorized approach.

    Args:
        eigenvector (np.ndarray): Eigenvectors (Ns, Nmodes).
        source_points (np.ndarray): Source points (Ns, 3).
        receiving_points (np.ndarray): Evaluation points (Nr, 3).
        k (float): Wavenumber.
        z_normalize (bool): Whether to apply sqrt(z) normalization.

    Returns:
        np.ndarray: Evaluated modes (Nmodes, Nr).
    """
    distances = cdist(receiving_points, source_points)  # (Nr, Ns)
    greens_kernel = np.exp(1j * k * distances) / distances
    projected = greens_kernel @ eigenvector
    scale_factor = (-1 / (4 * np.pi)) * np.sqrt(np.abs(receiving_points[:, 2])) if z_normalize else (-1 / (4 * np.pi))
    return (scale_factor * projected.T)

def evaluate_modes(eigenvector, source_points, receiving_points, k, z_normalize=True, method='direct'):
    """
    Dispatch function to evaluate modes using either direct or vectorized approach.

    Args:
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
    Plots the real part of the optical modes in a specified slice plane.

    Args:
        xx, yy, zz (np.ndarray): Meshgrid coordinate arrays.
        modes (np.ndarray): Modes evaluated at meshgrid points (Nmodes, ...).
        plane (str): Plane to plot ('xy', 'yz', or 'xz').
        x, y, z (int): Slice index to use for the specified plane.
    """
    modes = np.asarray([np.reshape(i, xx.shape) for i in modes])

    if plane == 'yz':
        if x is None:
            raise ValueError("For 'yz' plane, x-coordinate must be specified.")
        for mode in modes:
            sliced_modes = mode[x, :, :].squeeze()
            fig, ax = plt.subplots(figsize=(5,5))
            ax.imshow(np.real(sliced_modes), cmap='jet')
            plt.show()

    elif plane == 'xz':
        sliced_modes = modes[:, :, 0, :].squeeze()
        # Optional: Add plot here if needed

    elif plane == 'xy':
        if z is None:
            raise ValueError("For 'xy' plane, z-coordinate must be specified.")
        for i, mode in enumerate(modes):
            sliced_modes = mode[:, :, z].squeeze()
            fig, ax = plt.subplots(figsize=(5,5))
            fig.suptitle(f"Mode {i+1}")
            ax.imshow(np.real(sliced_modes), cmap='jet')
            plt.show()
    else:
        raise ValueError("Plane must be 'yz', 'xz', or 'xy'.")

def matprint(mat, fmt="g"):
    """
    Nicely prints a 2D matrix with aligned columns.

    Args:
        mat (np.ndarray): Matrix to print.
        fmt (str): Format string for each element.
    """
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for row in mat:
        for i, val in enumerate(row):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(val), end="  ")
        print("")

# ===================== MAIN EXECUTION BLOCK ===================== #

if __name__ == "__main__":
    # Simulation parameters
    wavelength = 1.0
    k = 2 * np.pi / wavelength
    Nsx, Nsy, dsx, dsy = 17, 17, wavelength, wavelength
    Nrx, Nry, drx, dry = 17, 17, wavelength, wavelength
    Lz = 50 * wavelength

    # Create 3D grid of sources and receivers
    s = np.asarray([(j * dsx, i * dsy, 0) for i in range(Nsy) for j in range(Nsx)])
    d = np.asarray([(j * drx, i * dry, Lz) for i in range(Nry) for j in range(Nrx)])

    # Compute transfer matrix
    Gsr = free_space_transfer_function(s, d, k, method='direct')

    # Compute normalization constant
    S = sum_rule(Gsr)
    print("S:", S)

    # Eigen-decomposition of Gsr^H Gsr

    Gsrd_Gsr = Gsr.conj().T @ Gsr
    eig_vect, eig_vals, Vh = np.linalg.svd(Gsrd_Gsr, full_matrices=True)
    #eig_vals, eig_vect = scipy.linalg.eigh(Gsr)
    #eig_vals = np.flip(eig_vals)
    #eig_vect = np.flip(eig_vect, axis=-1)
    eig_vect_normalized = normalize_eigenvector(eig_vect)
    
    print(np.max(eig_vals)/S)
    input()

    # Plot mode strengths
    plt.plot((eig_vals[0:20]/S)*100, marker='o')
    plt.title("Mode Energy Distribution")
    plt.xlabel("Mode Index")
    plt.ylabel("Percent Energy")
    plt.grid()
    plt.show()

    # Plot cumulative energy distribution
    plt.plot(np.cumsum(eig_vals)/S, marker='o')
    plt.xlabel("Mode Index")
    plt.ylabel("Cumulative Energy Fraction")
    plt.title("Cumulative Energy Fraction of Modes")
    plt.grid()
    plt.show()

    # Evaluate modes on a grid
    num_points = 300
    x = np.linspace(0, drx*Nrx, num_points)
    y = np.linspace(0, dry*Nry, num_points)
    z = Lz
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    evaluated_modes = evaluate_modes(eig_vect_normalized, s, points, k, method='vectorized', z_normalize=True)

    # Plot modes in the xy-plane at z=Lz
    plot_modes(xx, yy, zz, evaluated_modes, plane='xy', z=0)

