import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count

def scale_config(config: dict) -> dict:
    """
    Scales the parameters in the config dictionary by the wavelength.
    """
    wavelength = config.get('wavelength', 1.0)
    wavelength = float(wavelength)  # Ensure wavelength is a float
    if wavelength <= 0:
        raise ValueError("Wavelength must be a positive number.")
    
    # Scale the parameters in the config dictionary for the 'source' and 'receiver'
    for key in ['source', 'receiver']:
        Lx = config[key].get('Lx', None)
        Ly = config[key].get('Ly', None)
        Lz = config[key].get('Lz', None)
        center = config[key].get('center', (0, 0, 0))
        new_center = (np.round(center[0] / wavelength,5), 
                      np.round(center[1] / wavelength,5), 
                      np.round(center[2] / wavelength,5))
        config[key]['Lx'] = np.round(Lx / wavelength, 5) if Lx is not None else None
        config[key]['Ly'] = np.round(Ly / wavelength, 5) if Ly is not None else None
        config[key]['Lz'] = np.round(Lz / wavelength, 5) if Lz is not None else None
        config[key]['center'] = new_center
    # Scale the wavelength itself
    config['wavelength'] = wavelength / wavelength  # Normalize wavelength to 1
    
    return config

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
    distance = np.linalg.norm(s - d)
    return distance


def sum_rule(g: np.array, *args, **kwargs) -> float:
    """
    Computes the Frobenius norm squared (sum of squared magnitudes) of the transfer matrix.

    Args:
        g (np.ndarray): Transfer matrix.

    Returns:
        float: Frobenius norm squared.
    """
    return np.linalg.norm(g, 'fro') ** 2


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
    Normalize each column of an eigenvector matrix to have unit norm and phase-normalized center element.

    Args:
        eigenvector (np.ndarray): 2D array of eigenvectors (Ns, Nmodes).

    Returns:
        np.ndarray: Normalized eigenvectors.
    """
    print("Normalizing eigenvectors by phase and magnitude...")
    def normalize_column(vec):
        vec = normalize_phase(vec)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    return np.column_stack([normalize_column(eigenvector[:, j])
                            for j in range(eigenvector.shape[1])])


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
    print("Evaluating modes using direct method...")
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

def evaluate_chunk(eigenvector, source_points, receiving_points_chunk, k, z_normalize):
    distances = cdist(receiving_points_chunk, source_points)  # (Nr_chunk, Ns)
    greens_kernel = np.exp(1j * k * distances) / distances
    projected = greens_kernel @ eigenvector  # (Nr_chunk, Nmodes)
    if z_normalize:
        scale_factor = (-1 / (4 * np.pi)) * np.sqrt(np.abs(receiving_points_chunk[:, 2]))  # (Nr_chunk,)
        return (scale_factor[:, np.newaxis] * projected).T  # (Nmodes, Nr_chunk)
    else:
        scale_factor = -1 / (4 * np.pi)
        return (scale_factor * projected).T  # (Nmodes, Nr_chunk)

def evaluate_modes_parallel(eigenvector, source_points, receiving_points, k, z_normalize=True, max_workers=None, chunks=8):
    """
    Parallelized mode evaluation over receiving points.

    Args:
        eigenvector (np.ndarray): (Ns, Nmodes)
        source_points (np.ndarray): (Ns, 3)
        receiving_points (np.ndarray): (Nr, 3)
        k (float): Wavenumber
        z_normalize (bool): Normalize by sqrt(z)
        max_workers (int): CPUs to use (default: half available)
        chunks (int): Number of chunks to split receiving points into

    Returns:
        np.ndarray: Evaluated modes (Nmodes, Nr)
    """
    max_workers = max_workers or min(cpu_count() // 2, chunks)
    print("Evaluating modes in parallel (workers={})...".format(max_workers))
    Nr = receiving_points.shape[0]
    chunk_size = int(np.ceil(Nr / chunks))
    point_chunks = [receiving_points[i:i + chunk_size] for i in range(0, Nr, chunk_size)]

    results = Parallel(n_jobs=max_workers)(
        delayed(evaluate_chunk)(eigenvector, source_points, chunk, k, z_normalize)
        for chunk in point_chunks
    )

    return np.concatenate(results, axis=1)  # (Nmodes, Nr)

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
    print("Evaluating modes using vectorized approach...")
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
    elif method == 'parallel':
        return evaluate_modes_parallel(eigenvector, source_points, receiving_points, k, z_normalize)
    else:
        raise ValueError("Method not recognized. Use 'direct' or 'vectorized'.")

