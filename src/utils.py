import numpy as np
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
    Normalizes each column of an eigenvector matrix by phase.

    Args:
        eigenvector (np.ndarray): 2D array of eigenvectors.

    Returns:
        np.ndarray: Column-wise phase-normalized eigenvectors.
    """
    print("Normalizing eigenvectors by phase...")
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
    else:
        raise ValueError("Method not recognized. Use 'direct' or 'vectorized'.")

