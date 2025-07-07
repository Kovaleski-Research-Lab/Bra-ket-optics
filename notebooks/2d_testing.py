import numpy as np
import matplotlib.pyplot as plt
import scipy
import sympy
from scipy.spatial.distance import cdist  # fast distance computation
from scipy.linalg import eigh


def euclidean_distance(s, d) -> float:
    # s -> source point
    # d -> destination (receiving point)
    s = np.asarray(s)
    d = np.asarray(d)
    return np.linalg.norm(s - d)


def free_space_transfer_function_3d(r_s, r_r, k) -> np.ndarray:
    """
    Computes the 3D free-space Green's function between source and receiver points.

    Args:
        r_s (np.ndarray): Array of source points, shape (Ns, 3)
        r_r (np.ndarray): Array of receiver points, shape (Nr, 3)
        k (float): Wavenumber

    Returns:
        g (np.ndarray): Transfer function matrix of shape (Nr, Ns)
        check_d (float): Distance for a specific check case (optional debug)
    """
    r_s = np.asarray(r_s)
    r_r = np.asarray(r_r)
    g = np.zeros((len(r_r), len(r_s)), dtype=complex)

    check_d = None
    for i, r in enumerate(r_r):
        for j, s in enumerate(r_s):
            d = euclidean_distance(s, r)
            if j == 2 and i == 0:
                check_d = d
            numerator = -np.exp(1j * k * d)
            denominator = 4 * np.pi * d
            #g[i, j] = np.round(numerator / denominator, 5)  # rounding may be unnecessary for production
            g[i, j] = numerator / denominator
    return g, check_d

def free_space_transfer_function_3d_vectorized(r_s, r_r, k):
    """
    Vectorized computation of the 3D free-space Green's function.

    Args:
        r_s (np.ndarray): Source points of shape (Ns, 3)
        r_r (np.ndarray): Receiver points of shape (Nr, 3)
        k (float): Wavenumber

    Returns:
        g (np.ndarray): Transfer function matrix of shape (Nr, Ns)
    """
    # Ensure input shapes
    r_s = np.asarray(r_s)  # (Ns, 3)
    r_r = np.asarray(r_r)  # (Nr, 3)

    # Compute pairwise distances: ||r_r[i] - r_s[j]|| for all i, j
    # r_r[:, np.newaxis, :] shape: (Nr, 1, 3)
    # r_s[np.newaxis, :, :] shape: (1, Ns, 3)
    diffs = r_r[:, np.newaxis, :] - r_s[np.newaxis, :, :]  # shape: (Nr, Ns, 3)
    dists = np.linalg.norm(diffs, axis=-1)  # shape: (Nr, Ns)

    # Avoid division by zero (diagonal terms)
    dists = np.where(dists == 0, 1e-10, dists)

    # Compute Green's function: G = -exp(ikr) / (4πr)
    g = -np.exp(1j * k * dists)
    
    g *= -1.0 / (4 * np.pi * dists)

    return g

def compute_greens_blockwise(r_s, r_r, k, block_size=8000):
    Ns, Nr = len(r_s), len(r_r)
    g = np.zeros((Nr, Ns), dtype=np.complex64)
    for i in range(0, Nr, block_size):
        r_r_block = r_r[i:i+block_size]
        diffs = r_r_block[:, None, :] - r_s[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        dists = np.where(dists == 0, 1e-10, dists)
        g_block = -np.exp(1j * k * dists) / (4 * np.pi * dists)
        g[i:i+block_size] = g_block.astype(np.complex64)
    return g
\
# Choose the central phase to be zero, normalize by it
def normalize_phase(eigenvector):
    # Choose middle index (zero-based)
    mid_idx = len(eigenvector) // 2
    # Extract phase of the middle entry
    phase = np.angle(eigenvector[mid_idx])
    # Normalize eigenvector by this phase
    return eigenvector * np.exp(-1j * phase)

def sum_rule(g: np.ndarray) -> float:
    return np.linalg.norm(g, 'fro')**2


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")




if __name__ == "__main__":
    wavelength = 1
    Nsx = 17
    Nsy = 17
    ds = wavelength
    
    
    Lz = 50 * wavelength
    Nrx = 17
    Nry = 17
    k = 2 * np.pi / wavelength


    sx = np.linspace(0, (Nsx * wavelength), Nsx)
    sy = np.linspace(0, (Nsy * wavelength), Nsy)
    Sx, Sy = np.meshgrid(sx, sy)
    Sx = Sx.ravel()
    Sy = Sy.ravel()
    Sz = np.full_like(Sx, 0)
    source_points = np.stack([Sx, Sy, Sz], axis=1)
    
    
    rx = np.linspace(0, (Nrx * wavelength), Nrx)
    ry = np.linspace(0, (Nry * wavelength), Nry)
    Rx, Ry = np.meshgrid(rx, ry)
    Rx = Rx.ravel()
    Ry = Ry.ravel()
    Rz = np.full_like(Rx, Lz)
    receiver_points = np.stack([Rx, Ry, Rz], axis=1)

    
    Gsr = compute_greens_blockwise(source_points, receiver_points, k)

    S = sum_rule(Gsr)

    print(S)

    Gsr_Gsrd = np.matmul(np.matrix.getH(Gsr), Gsr)
    eig_vals, eig_vect = np.linalg.eigh(Gsr_Gsrd)
    eig_vect = np.flip(eig_vect, axis=1)
    eig_vals = np.flip(eig_vals)
    plt.plot(eig_vals[0:20])

    # Normalize each eigenvector's phase
    normalized_eigenvectors = np.column_stack([
        normalize_phase(eig_vect[:, j])
        for j in range(eig_vect.shape[1])
    ])


    # Destination points is just a higher resolution receiving plane
    dest_x = np.linspace(0, (Nrx * wavelength), 100)
    dest_y = np.linspace(0, (Nry * wavelength), 100)
    Dest_x, Dest_y = np.meshgrid(dest_x, dest_y)
    Dest_x = Dest_x.ravel()
    Dest_y = Dest_y.ravel()
    Dest_z = np.full_like(Dest_x, Lz)
    dest_points = np.stack([Dest_x,Dest_y, Dest_z], axis=1)


    # Convert to arrays if they aren't already
    source_points = np.asarray(source_points)
    dest_points = np.asarray(dest_points)
    normalized_eigenvectors = np.asarray(normalized_eigenvectors)
    
    # Precompute all pairwise distances: shape (num_dest, num_source)
    distances = cdist(dest_points, source_points)  # much faster than nested loops
    
    # Precompute Green's function kernel (excluding eigenvector weighting)
    greens_kernel = np.exp(1j * k * distances) / distances  # shape: (num_dest, num_source)
    
    # Apply the integral and eigenvector projection
    # Transpose normalized_eigenvectors to shape (num_modes, num_source)
    projected = greens_kernel @ normalized_eigenvectors.T  # shape: (num_dest, num_modes)
    
    # Multiply by constants and reshape to match original output (num_modes, num_dest)
    scale_factor = (-1 / (4 * np.pi)) * np.sqrt(Lz)
    values = scale_factor * projected.T  # shape: (num_modes, num_dest)


    values = values.reshape(len(normalized_eigenvectors), 100, 100)

    # Plot the first mode
    plt.imshow(values[0].real)