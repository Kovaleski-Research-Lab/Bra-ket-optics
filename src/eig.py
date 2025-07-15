import numpy as np
from tqdm import tqdm
import scipy.linalg
from scipy.sparse.linalg import svds
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from .utils import euclidean_distance, normalize_eigenvector

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
            d = euclidean_distance(s_point, r_point)
            numerator = -np.exp(1j * k * d)
            denominator = 4 * np.pi * d
            g[i, j] = np.round(numerator / denominator, 5)
    return g

def _compute_block(args):
    """
    Helper function to compute one block of the transfer function.
    """
    s, r_block, k, block_start = args
    diffs = r_block[:, None, :] - s[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    dists = np.where(dists == 0, np.finfo(float).eps, dists)
    g_block = -np.exp(1j * k * dists) / (4 * np.pi * dists)
    return block_start, g_block

def blockwise_transfer_function(s, r, k, blocksize=500, **kwargs) -> np.ndarray:
    """
    Computes the transfer function in parallel blocks for memory efficiency and speed.

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

    blocks = [(s, r[i:i + blocksize], k, i) for i in range(0, Nr, blocksize)]
    num_blocks = len(blocks)

    max_cpus = multiprocessing.cpu_count()
    num_workers = min(max_cpus // 2, num_blocks)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(_compute_block, blocks), total=num_blocks,
                            desc=f"Parallel transfer function (workers={num_workers})"))

    for block_start, g_block in results:
        g[block_start:block_start + g_block.shape[0], :] = g_block

    return g

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


def calculate_modes(Gsr, normalize=True, max_components=None):
    """
    Calculate modes from the transfer function matrix Gsr.
    
    Args:
        Gsr (np.ndarray): Transfer function matrix (Nr, Ns).
        k (float): Wavenumber.
        normalize (bool): Whether to normalize the eigenvectors.
    Returns:
        eig_vals (np.ndarray): Eigenvalues of Gsr^† Gsr.
        eig_vect_normalized (np.ndarray): Normalized eigenvectors.
    """
    print("Calculating modes...")
    # Eigen-decomposition of Gsr^H Gsr
    Gsrd_Gsr = Gsr.conj().T @ Gsr
    if max_components is None:
        eig_vect, eig_vals, Vh = scipy.linalg.svd(Gsrd_Gsr, full_matrices=True)
    else:
        eig_vect, eig_vals, Vh = svds(Gsrd_Gsr, k=max_components)
        eig_vals = np.flip(eig_vals)
        eig_vect = np.flip(eig_vect, axis=-1)
    if normalize:
        eig_vect_normalized = normalize_eigenvector(eig_vect)
    else:
        eig_vect_normalized = eig_vect
    return eig_vals, eig_vect_normalized
