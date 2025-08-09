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
        results = list(executor.map(_compute_block, blocks))

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


def calculate_modes(Gsr, normalize=True, max_components=None, verbose=False):
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

    if verbose:
        print("Calculating modes...")

    if max_components is None:
        # SVD on Gsr to get the right and left singular vectors
        U, s, Vh = np.linalg.svd(Gsr, full_matrices=True)
        # Sort based on |s|^2
        idx = np.argsort(np.abs(s)**2)[::-1]
        U = U[:, idx]
        s = s[idx]
        Vh = Vh[idx, :]

        # Conjugate transpose Vh
        Vh = Vh.conj().T  # Make Vh a column matrix

    else:
        # Use sparse SVD for large matrices
        U, s, Vh = svds(Gsr, k=max_components)
        # Sort based on |s|^2
        idx = np.argsort(np.abs(s)**2)[::-1]
        U = U[:, idx]
        s = s[idx]
        Vh = Vh[idx, :]

        # Conjugate transpose Vh
        Vh = Vh.conj().T

    assert np.allclose(Gsr, U @ np.diag(s) @ Vh.conj().T), "SVD reconstruction failed before normalization"

    if normalize:
        # Normalize the singular vectors
        U = normalize_eigenvector(U)
        Vh = normalize_eigenvector(Vh)
        assert np.allclose(np.linalg.norm(U, axis=0), 1.0)
        assert np.allclose(np.linalg.norm(Vh, axis=0), 1.0)

    eig_vect_source = Vh
    eig_vect_receiver = U
    eig_vals = s

    return eig_vect_receiver, eig_vals, eig_vect_source


def forward_projection_direct(source_field:np.ndarray, Gsr:np.ndarray) -> np.ndarray:
    """
    Projects the source field onto the receiver points using the transfer function matrix Gsr.

    Args:
        source_field (np.ndarray): Field at source points (Ns,).
        Gsr (np.ndarray): Transfer function matrix (Nr, Ns).

    Returns:
        np.ndarray: Projected field at receiver points (Nr,).
    """
    if source_field.ndim == 2 and source_field.shape[1] != 1:
        source_field = source_field.flatten()
    if source_field.ndim == 1:
        source_field = source_field.reshape(-1, 1)

    return Gsr @ source_field 


def inverse_projection_direct(received_field:np.ndarray, Gsr:np.ndarray) -> np.ndarray:
    """
    Projects the received field back to the source points using the transfer function matrix Gsr.

    Args:
        received_field (np.ndarray): Field at receiver points (Nr,).
        Gsr (np.ndarray): Transfer function matrix (Nr, Ns).

    Returns:
        np.ndarray: Projected field at source points (Ns,).
    """
    if received_field.ndim == 2 and received_field.shape[1] != 1:
        received_field = received_field.flatten()
    if received_field.ndim == 1:
        received_field = received_field.reshape(-1, 1)

    return np.linalg.pinv(Gsr) @ received_field

def forward_projection_eig(source_field: np.ndarray, eig_vals: np.ndarray,
                           eig_vect_source: np.ndarray, eig_vect_receiver: np.ndarray) -> np.ndarray:
    """
    Perform forward propagation using mode decomposition:
    |phi_Ro> = sum_j (1 / s_j*) <psi_Sj|psi_Si> |phi_Rj>

    Args:
        source_field (np.ndarray): Source field (Ns, 1).
        eig_vals (np.ndarray): Eigenvalues s_j.
        eig_vect_source (np.ndarray): Right singular vectors (Ns, Nmodes).
        eig_vect_receiver (np.ndarray): Left singular vectors (Nr, Nmodes).

    Returns:
        np.ndarray: Receiver field (Nr, 1).
    """
    # Ensure shapes
    assert source_field.ndim == 2 and source_field.shape[1] == 1, "source_field must be a (Ns, 1) column vector"

    # <psi_Sj|psi_Si>: inner product of each source mode with the input
    a_j = eig_vect_source.conj().T @ source_field  # shape: (Nmodes, 1)

    # Divide by s_j*
    scale = a_j * np.conj(eig_vals).reshape(-1, 1)  # shape: (Nmodes, 1)

    # Sum_j (a_j / s_j*) * |phi_Rj>
    output = eig_vect_receiver @ scale  # shape: (Nr, 1)

    return output


def inverse_projection_eig(receiver_field: np.ndarray, eig_vals: np.ndarray,
                           eig_vect_source: np.ndarray, eig_vect_receiver: np.ndarray) -> np.ndarray:
    """
    Inverse propagation using mode decomposition.
    
    Args:
        receiver_field (np.ndarray): Complex receiver field vector of shape (Nr,).
        eig_vals (np.ndarray): Singular values s_j of shape (Nmodes,).
        eig_vect_source (np.ndarray): Source eigenvectors ψ_Sj of shape (Ns, Nmodes).
        eig_vect_receiver (np.ndarray): Receiver eigenvectors ϕ_Rj of shape (Nr, Nmodes).

    Returns:
        np.ndarray: Source field vector of shape (Ns,).
    """
    # Ensure shapes
    assert receiver_field.ndim == 2 and receiver_field.shape[1] == 1, "receiver_field must be a (Nr, 1) column vector"

    # Project receiver field onto receiver modes
    a_j = np.conj(eig_vect_receiver.T) @ receiver_field  # Shape: (Nmodes,)
    
    # Weight by 1 / s_j
    weights = a_j / eig_vals.reshape(-1,1)  # Shape: (Nmodes,)
    
    # Sum over weighted source modes
    source_field = eig_vect_source @ weights  # Shape: (Ns,)
    
    return source_field

