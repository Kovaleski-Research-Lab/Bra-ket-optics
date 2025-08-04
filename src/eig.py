import numpy as np
from tqdm import tqdm
import scipy.linalg
from scipy.sparse.linalg import svds
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from .utils import euclidean_distance, normalize_eigenvector

multiprocessing.set_start_method('spawn', force=True)

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
    g = np.zeros((len(r), len(s)), dtype=np.complex128)
    for i, r_point in tqdm(enumerate(r), desc="Computing transfer function (direct method)", total=len(r)):
        for j, s_point in enumerate(s):
            d = euclidean_distance(s_point, r_point)
            numerator = -np.exp(1j * k * d)
            denominator = 4 * np.pi * d
            g[i, j] = numerator / denominator
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
    g = np.zeros((Nr, Ns), dtype=np.complex128)

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
        return blockwise_transfer_function(s, r, k, blocksize)
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
        eig_vect_receiver (np.ndarray): Eigenvectors of the receiver modes.
        eig_vals (np.ndarray): Eigenvalues of the source modes.
        eig_vect_source (np.ndarray): Eigenvectors of the source modes.
    """
    print("Calculating modes...")

    if max_components is None:
        # SVD on Gsr to get the right and left singular vectors
        U, s, Vh = np.linalg.svd(Gsr, full_matrices=True)
        # Sort based on |s|^2
        idx = np.argsort(np.abs(s)**2)[::-1]
        U = U[:, idx]
        s = s[idx]
        Vh = Vh[:, idx]

        # Conjugate transpose Vh
        Vh = Vh.conj().T  # Make Vh a column matrix

    else:
        # Use sparse SVD for large matrices
        U, s, Vh = svds(Gsr, k=max_components)
        # Sort based on |s|^2
        idx = np.argsort(np.abs(s)**2)[::-1]
        U = U[:, idx]
        s = s[idx]
        Vh = Vh[:, idx]

        # Conjugate transpose Vh
        Vh = Vh.conj().T

    if normalize:
        # Normalize the singular vectors
        U = normalize_eigenvector(U)
        Vh = normalize_eigenvector(Vh)

    eig_vect_source = Vh
    eig_vect_receiver = U
    eig_vals = s

    return eig_vect_receiver, eig_vals, eig_vect_source


def forward_projection(source_field:np.ndarray, Gsr:np.ndarray) -> np.ndarray:
    """
    Projects the source field onto the receiver points using the transfer function matrix Gsr.

    Args:
        source_field (np.ndarray): Field at source points (Ns,).
        Gsr (np.ndarray): Transfer function matrix (Nr, Ns).

    Returns:
        np.ndarray: Projected field at receiver points (Nr,).
    """
    return Gsr @ source_field 


def inverse_projection(aj:np.ndarray, sj:np.ndarray, receiver_eig_vect:np.ndarray) -> np.ndarray:
    """
    Projects the coefficients aj back to the source points using the receiver eigenvectors.

    Args:
        aj (np.ndarray): Coefficients at receiver points (Nr,).
        sj (np.ndarray): Source eigenvectors (Ns, Nr).
        receiver_eig_vect (np.ndarray): Eigenvectors of the receiver modes (Nr, Nr).

    Returns:
        np.ndarray: Projected field at source points (Ns,).
    """
    weights = aj / sj
    return receiver_eig_vect.T @ weights
