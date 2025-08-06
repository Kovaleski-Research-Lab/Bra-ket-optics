import numpy as np
import pytest
from src.eig import (
    forward_projection_direct,
    inverse_projection_direct,
    free_space_transfer_function,
    calculate_modes,
    forward_projection_eig,
    inverse_projection_eig
)
from src.geometry import create_points
import logging
import os

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Set up logger
logger = logging.getLogger("physics_sim")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("logs/test_projections.log", mode='w')  # Overwrite each test run
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(fh)

@pytest.fixture
def test_config_01():
    return {
        'source': {
            'geometry': 'plane',
            'axis': 'xy',
            'Nx': 4,
            'Ny': 4,
            'Nz': None,
            'Lx': 4,
            'Ly': 4,
            'Lz': None,
            'center': [0, 0, 0]
        },
        'receiver': {
            'geometry': 'plane',
            'axis': 'xy',
            'Nx': 4,
            'Ny': 4,
            'Nz': None,
            'Lx': 4,
            'Ly': 4,
            'Lz': None,
            'center': [0, 0, 10]
        }
    }

@pytest.fixture
def test_config_02():
    return {
        'source': {
            'geometry': 'plane',
            'axis': 'xy',
            'Nx': 2,
            'Ny': 2,
            'Nz': None,
            'Lx': 2,
            'Ly': 2,
            'Lz': None,
            'center': [0, 0, 0]
        },
        'receiver': {
            'geometry': 'plane',
            'axis': 'xy',
            'Nx': 2,
            'Ny': 2,
            'Nz': None,
            'Lx': 2,
            'Ly': 2,
            'Lz': None,
            'center': [0, 0, 10]
        }
    }

@pytest.fixture
def source_field_2x2():
    return generate_source_field_2x2()

@pytest.fixture
def source_field_4x4():
    return generate_source_field_4x4()

@pytest.fixture
def source_field_4x4_gaussian():
    return create_gaussian(4, 4, sigma=0.3)
    

def generate_source_field_4x4():
    return np.array([[1+1j, 2+2j, 3+3j, 4+4j],
                     [5+5j, 6+6j, 7+7j, 8+8j],
                     [9+9j, 10+10j, 11+11j, 12+12j],
                     [13+13j, 14+14j, 15+15j, 16+16j]])

def generate_source_field_2x2():
    return np.array([[1+1j, 2+2j],
                     [3+3j, 4+4j]])

def create_gaussian(nx: int, ny: int, sigma: float = 0.3) -> np.ndarray:
        """Create Gaussian field pattern."""
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        return np.exp(-(X**2 + Y**2) / (2 * sigma**2)) * (1 + 1j)

def forward_by_hand(Gsr, source_field):
    """
    Manually perform forward projection via matrix multiplication.
    Args:
        Gsr (np.ndarray): Transfer function (Nr, Ns).
        source_field (np.ndarray): Complex source field (2D grid or already flattened).
    Returns:
        np.ndarray: Projected field at receivers (Nr, 1).
    """
    projected_field = np.zeros((Gsr.shape[0], 1), dtype=np.complex128)
    if source_field.ndim == 2:
        # Flatten the source field if it's a 2D grid
        source_field = source_field.flatten()
    # For each receiver point, compute the sum of contributions from all source points
    for i in range(Gsr.shape[0]):
        for j in range(Gsr.shape[1]):
            projected_field[i] += Gsr[i, j] * source_field[j]
    return projected_field

def inverse_by_hand(Gsr, received_field):
    """
    Manually perform inverse projection using the Moore-Penrose pseudoinverse.
    Args:
        Gsr (np.ndarray): Transfer function (Nr, Ns).
        received_field (np.ndarray): Complex field at receivers (Nr, 1).
    Returns:
        np.ndarray: Reconstructed source field (Ns, 1).
    """
    # Compute the pseudoinverse of Gsr
    Gsr_pinv = np.linalg.pinv(Gsr)
    if received_field.ndim == 2 and received_field.shape[1] != 1:
        received_field = received_field.flatten()
    if received_field.ndim == 1:
        received_field = received_field.reshape(-1, 1)
    
    # Perform the projection using for loops
    reconstructed_source = np.zeros((Gsr.shape[1], 1), dtype=np.complex128)
    for i in range(Gsr.shape[1]):
        for j in range(Gsr.shape[0]):
            reconstructed_source[i] += Gsr_pinv[j, i] * received_field[j]
    return reconstructed_source

# ============ FORWARD PROJECTION TESTS - Cross-Method Comparisons ============


@pytest.mark.parametrize("rtol", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
@pytest.mark.parametrize("atol", [1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
def test_forward_all_methods_agree_2x2(test_config_02, source_field_2x2, rtol, atol):
    """Test that all three forward methods return identical results for 2x2 case."""
    logger.debug("==== Testing forward method agreement 2x2 ====")
    config = test_config_02
    source_points = create_points(config['source'])
    receiving_points = create_points(config['receiver'])
    k = 2 * np.pi
    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
    
    # Prepare source field (normalized for eig method)
    source_vector = source_field_2x2.reshape(-1, 1)
    source_vector = source_vector / np.linalg.norm(source_vector)
    
    # Get results from all three methods
    result_by_hand = forward_by_hand(Gsr, source_vector)
    result_direct = forward_projection_direct(source_vector, Gsr)
    result_eig = forward_projection_eig(source_vector, eig_vals, eig_vect_source, eig_vect_receiver)
    
    logger.debug(f"By hand result shape: {result_by_hand.shape}")
    logger.debug(f"Direct result shape: {result_direct.shape}")
    logger.debug(f"Eig result shape: {result_eig.shape}")
    
    # All methods should return the same shape
    assert result_by_hand.shape == result_direct.shape == result_eig.shape
    
    # Compare all methods (note: eig method uses normalized input, so scale comparison appropriately)
    np.testing.assert_allclose(result_by_hand, result_direct, rtol=rtol, atol=atol)
    # For eig comparison, we need to account for the normalization
    expected_eig_scale = np.linalg.norm(source_vector)
    np.testing.assert_allclose(result_by_hand, result_eig * expected_eig_scale, rtol=rtol, atol=atol)

@pytest.mark.parametrize("rtol", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
@pytest.mark.parametrize("atol", [1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
def test_forward_all_methods_agree_4x4(test_config_01, source_field_4x4, rtol, atol):
    """Test that all three forward methods return identical results for 4x4 case."""
    logger.debug("==== Testing forward method agreement 4x4 ====")
    config = test_config_01
    source_points = create_points(config['source'])
    receiving_points = create_points(config['receiver'])
    k = 2 * np.pi
    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
    
    # Prepare source field (normalized for eig method)
    source_vector = source_field_4x4.reshape(-1, 1)
    source_vector = source_vector / np.linalg.norm(source_vector)
    
    # Get results from all three methods
    result_by_hand = forward_by_hand(Gsr, source_vector)
    result_direct = forward_projection_direct(source_vector, Gsr)
    result_eig = forward_projection_eig(source_vector, eig_vals, eig_vect_source, eig_vect_receiver)
    
    logger.debug(f"By hand result shape: {result_by_hand.shape}")
    logger.debug(f"Direct result shape: {result_direct.shape}")
    logger.debug(f"Eig result shape: {result_eig.shape}")
    
    # All methods should return the same shape
    assert result_by_hand.shape == result_direct.shape == result_eig.shape
    
    # Compare all methods
    np.testing.assert_allclose(result_by_hand, result_direct, rtol=rtol, atol=atol)
    # For eig comparison, we need to account for the normalization
    expected_eig_scale = np.linalg.norm(source_vector)
    np.testing.assert_allclose(result_by_hand, result_eig * expected_eig_scale, rtol=rtol, atol=atol)

@pytest.mark.parametrize("rtol", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
@pytest.mark.parametrize("atol", [1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
def test_forward_all_methods_agree_4x4_gaussian(test_config_01, source_field_4x4_gaussian, rtol, atol):
    """Test that all three forward methods return identical results for 4x4 case."""
    logger.debug("==== Testing forward method agreement 4x4 ====")
    config = test_config_01
    source_points = create_points(config['source'])
    receiving_points = create_points(config['receiver'])
    k = 2 * np.pi
    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
    
    # Prepare source field (normalized for eig method)
    source_vector = source_field_4x4_gaussian.reshape(-1, 1)
    source_vector = source_vector / np.linalg.norm(source_vector)
    
    # Get results from all three methods
    result_by_hand = forward_by_hand(Gsr, source_vector)
    result_direct = forward_projection_direct(source_vector, Gsr)
    result_eig = forward_projection_eig(source_vector, eig_vals, eig_vect_source, eig_vect_receiver)
    
    logger.debug(f"By hand result shape: {result_by_hand.shape}")
    logger.debug(f"Direct result shape: {result_direct.shape}")
    logger.debug(f"Eig result shape: {result_eig.shape}")
    
    # All methods should return the same shape
    assert result_by_hand.shape == result_direct.shape == result_eig.shape
    
    # Compare all methods
    np.testing.assert_allclose(result_by_hand, result_direct, rtol=rtol, atol=atol)
    # For eig comparison, we need to account for the normalization
    expected_eig_scale = np.linalg.norm(source_vector)
    np.testing.assert_allclose(result_by_hand, result_eig * expected_eig_scale, rtol=rtol, atol=atol)

# ============ INVERSE PROJECTION TESTS - Cross-Method Comparisons ============

@pytest.mark.parametrize("rtol", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
@pytest.mark.parametrize("atol", [1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
def test_inverse_all_methods_agree_2x2(test_config_02, source_field_2x2, rtol, atol):
    """Test that all three inverse methods return identical results for 2x2 case."""
    logger.debug("==== Testing inverse method agreement 2x2 ====")
    config = test_config_02
    source_points = create_points(config['source'])
    receiving_points = create_points(config['receiver'])
    k = 2 * np.pi
    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
    
    # Create a projected field to invert (using forward_by_hand as reference)
    source_vector = source_field_2x2.reshape(-1, 1)
    source_vector = source_vector / np.linalg.norm(source_vector)
    projected_field = forward_by_hand(Gsr, source_vector)
    
    # Get results from all three inverse methods
    result_by_hand = inverse_by_hand(Gsr, projected_field)
    result_direct = inverse_projection_direct(projected_field, Gsr)
    result_eig = inverse_projection_eig(projected_field, eig_vals, eig_vect_source, eig_vect_receiver)
    
    logger.debug(f"Original source shape: {source_vector.shape}")
    logger.debug(f"By hand result shape: {result_by_hand.shape}")
    logger.debug(f"Direct result shape: {result_direct.shape}")
    logger.debug(f"Eig result shape: {result_eig.shape}")
    
    # All methods should return the same shape
    assert result_by_hand.shape == result_direct.shape == result_eig.shape
    
    # Compare all methods
    np.testing.assert_allclose(result_by_hand, result_direct, rtol=rtol, atol=atol)
    np.testing.assert_allclose(result_by_hand, result_eig, rtol=rtol, atol=atol)
    np.testing.assert_allclose(result_direct, result_eig, rtol=rtol, atol=atol)

@pytest.mark.parametrize("rtol", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
@pytest.mark.parametrize("atol", [1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
def test_inverse_all_methods_agree_4x4(test_config_01, source_field_4x4, rtol, atol):
    """Test that all three inverse methods return identical results for 4x4 case."""
    logger.debug("==== Testing inverse method agreement 4x4 ====")
    config = test_config_01
    source_points = create_points(config['source'])
    receiving_points = create_points(config['receiver'])
    k = 2 * np.pi
    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
    
    # Create a projected field to invert (using forward_by_hand as reference)
    source_vector = source_field_4x4.reshape(-1, 1)
    source_vector = source_vector / np.linalg.norm(source_vector)
    projected_field = forward_by_hand(Gsr, source_vector)
    
    # Get results from all three inverse methods
    result_by_hand = inverse_by_hand(Gsr, projected_field)
    result_direct = inverse_projection_direct(projected_field, Gsr)
    result_eig = inverse_projection_eig(projected_field, eig_vals, eig_vect_source, eig_vect_receiver)
    
    logger.debug(f"Original source shape: {source_vector.shape}")
    logger.debug(f"By hand result shape: {result_by_hand.shape}")
    logger.debug(f"Direct result shape: {result_direct.shape}")
    logger.debug(f"Eig result shape: {result_eig.shape}")
    
    # All methods should return the same shape
    assert result_by_hand.shape == result_direct.shape == result_eig.shape
    
    # Compare all methods
    np.testing.assert_allclose(result_by_hand, result_direct, rtol=rtol, atol=atol)
    np.testing.assert_allclose(result_by_hand, result_eig, rtol=rtol, atol=atol)
    np.testing.assert_allclose(result_direct, result_eig, rtol=rtol, atol=atol)

@pytest.mark.parametrize("rtol", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
@pytest.mark.parametrize("atol", [1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
def test_inverse_all_methods_agree_4x4_gaussian(test_config_01, source_field_4x4_gaussian, rtol, atol):
    """Test that all three inverse methods return identical results for 4x4 case."""
    logger.debug("==== Testing inverse method agreement 4x4 ====")
    config = test_config_01
    source_points = create_points(config['source'])
    receiving_points = create_points(config['receiver'])
    k = 2 * np.pi
    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
    
    # Create a projected field to invert (using forward_by_hand as reference)
    source_vector = source_field_4x4_gaussian.reshape(-1, 1)
    source_vector = source_vector / np.linalg.norm(source_vector)
    projected_field = forward_by_hand(Gsr, source_vector)
    
    # Get results from all three inverse methods
    result_by_hand = inverse_by_hand(Gsr, projected_field)
    result_direct = inverse_projection_direct(projected_field, Gsr)
    result_eig = inverse_projection_eig(projected_field, eig_vals, eig_vect_source, eig_vect_receiver)
    
    logger.debug(f"Original source shape: {source_vector.shape}")
    logger.debug(f"By hand result shape: {result_by_hand.shape}")
    logger.debug(f"Direct result shape: {result_direct.shape}")
    logger.debug(f"Eig result shape: {result_eig.shape}")
    
    # All methods should return the same shape
    assert result_by_hand.shape == result_direct.shape == result_eig.shape
    
    # Compare all methods
    np.testing.assert_allclose(result_by_hand, result_direct, rtol=rtol, atol=atol)
    np.testing.assert_allclose(result_by_hand, result_eig, rtol=rtol, atol=atol)
    np.testing.assert_allclose(result_direct, result_eig, rtol=rtol, atol=atol)

# ============ ROUND-TRIP TESTS ============

@pytest.mark.parametrize("rtol", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
@pytest.mark.parametrize("atol", [1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
def test_forward_inverse_round_trip_all_methods_2x2(test_config_02, source_field_2x2, rtol, atol):
    """Test forward->inverse round trip for all method combinations on 2x2."""
    logger.debug("==== Testing round trip all methods 2x2 ====")
    config = test_config_02
    source_points = create_points(config['source'])
    receiving_points = create_points(config['receiver'])
    k = 2 * np.pi
    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
    
    source_vector = source_field_2x2.reshape(-1, 1)
    source_vector = source_vector / np.linalg.norm(source_vector)
    
    # Test all 9 combinations of forward/inverse methods
    forward_methods = [
        ("by_hand", lambda s: forward_by_hand(Gsr, s.reshape(source_vector.shape))),
        ("direct", lambda s: forward_projection_direct(s, Gsr)),
        ("eig", lambda s: forward_projection_eig(s/np.linalg.norm(s), eig_vals, eig_vect_source, eig_vect_receiver) * np.linalg.norm(s))
    ]
    
    inverse_methods = [
        ("by_hand", lambda p: inverse_by_hand(Gsr, p)),
        ("direct", lambda p: inverse_projection_direct(p, Gsr)),
        ("eig", lambda p: inverse_projection_eig(p, eig_vals, eig_vect_source, eig_vect_receiver))
    ]
    
    for fwd_name, fwd_func in forward_methods:
        for inv_name, inv_func in inverse_methods:
            logger.debug(f"Testing {fwd_name} -> {inv_name}")
            
            # Forward projection
            projected = fwd_func(source_vector)
            
            # Inverse projection
            recovered = inv_func(projected)
            
            # Check round trip (allowing for some numerical error and potential scaling)
            # Normalize both to compare shapes and relative values
            #source_normalized = source_vector / np.linalg.norm(source_vector)
            #recovered_normalized = recovered / np.linalg.norm(recovered)
            
            assert recovered.shape == source_vector.shape, \
                f"{fwd_name}->{inv_name}: Shape mismatch {recovered.shape} != {source_vector.shape}"
            
            # Check if they are close (either directly or with a phase factor)
            try:
                np.testing.assert_allclose(source_vector, recovered, rtol=rtol, atol=atol)
                logger.debug(f"{fwd_name}->{inv_name}: PASSED (direct match)")
            except AssertionError:
                # Try with phase alignment
                phase_factor = np.vdot(recovered, source_vector) / np.vdot(source_vector, source_vector)
                phase_factor = phase_factor / abs(phase_factor)  # Normalize to unit magnitude
                try:
                    np.testing.assert_allclose(source_vector, recovered * phase_factor, rtol=rtol, atol=atol)
                    logger.debug(f"{fwd_name}->{inv_name}: PASSED (with phase correction)")
                except AssertionError:
                    logger.error(f"{fwd_name}->{inv_name}: FAILED - no good alignment found")
                    raise

@pytest.mark.parametrize("rtol", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
@pytest.mark.parametrize("atol", [1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
def test_forward_inverse_round_trip_all_methods_4x4(test_config_01, source_field_4x4, rtol, atol):
    """Test forward->inverse round trip for all method combinations on 4x4."""
    logger.debug("==== Testing round trip all methods 4x4 ====")
    config = test_config_01
    source_points = create_points(config['source'])
    receiving_points = create_points(config['receiver'])
    k = 2 * np.pi
    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
    
    source_vector = source_field_4x4.reshape(-1, 1)
    source_vector = source_vector / np.linalg.norm(source_vector)
    
    # Test all 9 combinations of forward/inverse methods
    forward_methods = [
        ("by_hand", lambda s: forward_by_hand(Gsr, s.reshape(source_vector.shape))),
        ("direct", lambda s: forward_projection_direct(s, Gsr)),
        ("eig", lambda s: forward_projection_eig(s/np.linalg.norm(s), eig_vals, eig_vect_source, eig_vect_receiver) * np.linalg.norm(s))
    ]
    
    inverse_methods = [
        ("by_hand", lambda p: inverse_by_hand(Gsr, p)),
        ("direct", lambda p: inverse_projection_direct(p, Gsr)),
        ("eig", lambda p: inverse_projection_eig(p, eig_vals, eig_vect_source, eig_vect_receiver))
    ]
    
    for fwd_name, fwd_func in forward_methods:
        for inv_name, inv_func in inverse_methods:
            logger.debug(f"Testing {fwd_name} -> {inv_name}")
            
            # Forward projection
            projected = fwd_func(source_vector)
            
            # Inverse projection
            recovered = inv_func(projected)
            
            # Check round trip (allowing for some numerical error and potential scaling)
            # Normalize both to compare shapes and relative values
            #source_normalized = source_vector / np.linalg.norm(source_vector)
            #recovered_normalized = recovered / np.linalg.norm(recovered)
            
            assert recovered.shape == source_vector.shape, \
                f"{fwd_name}->{inv_name}: Shape mismatch {recovered.shape} != {source_vector.shape}"
            
            # Check if they are close (either directly or with a phase factor)
            try:
                np.testing.assert_allclose(source_vector, recovered, rtol=rtol, atol=atol)
                logger.debug(f"{fwd_name}->{inv_name}: PASSED (direct match)")
            except AssertionError:
                # Try with phase alignment
                phase_factor = np.vdot(recovered, source_vector) / np.vdot(source_vector, source_vector)
                phase_factor = phase_factor / abs(phase_factor)  # Normalize to unit magnitude
                try:
                    np.testing.assert_allclose(source_vector, recovered * phase_factor, rtol=rtol, atol=atol)
                    logger.debug(f"{fwd_name}->{inv_name}: PASSED (with phase correction)")
                except AssertionError:
                    logger.error(f"{fwd_name}->{inv_name}: FAILED - no good alignment found")
                    raise

@pytest.mark.parametrize("rtol", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
@pytest.mark.parametrize("atol", [1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
def test_forward_inverse_round_trip_all_methods_4x4_gaussian(test_config_01, source_field_4x4_gaussian, rtol, atol):
    """Test forward->inverse round trip for all method combinations on 4x4."""
    logger.debug("==== Testing round trip all methods 4x4 ====")
    config = test_config_01
    source_points = create_points(config['source'])
    receiving_points = create_points(config['receiver'])
    k = 2 * np.pi
    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
    
    source_vector = source_field_4x4_gaussian.reshape(-1, 1)
    source_vector = source_vector / np.linalg.norm(source_vector)
    
    # Test all 9 combinations of forward/inverse methods
    forward_methods = [
        ("by_hand", lambda s: forward_by_hand(Gsr, s.reshape(source_vector.shape))),
        ("direct", lambda s: forward_projection_direct(s, Gsr)),
        ("eig", lambda s: forward_projection_eig(s/np.linalg.norm(s), eig_vals, eig_vect_source, eig_vect_receiver) * np.linalg.norm(s))
    ]
    
    inverse_methods = [
        ("by_hand", lambda p: inverse_by_hand(Gsr, p)),
        ("direct", lambda p: inverse_projection_direct(p, Gsr)),
        ("eig", lambda p: inverse_projection_eig(p, eig_vals, eig_vect_source, eig_vect_receiver))
    ]
    
    for fwd_name, fwd_func in forward_methods:
        for inv_name, inv_func in inverse_methods:
            logger.debug(f"Testing {fwd_name} -> {inv_name}")
            
            # Forward projection
            projected = fwd_func(source_vector)
            
            # Inverse projection
            recovered = inv_func(projected)
            
            # Check round trip (allowing for some numerical error and potential scaling)
            # Normalize both to compare shapes and relative values
            #source_normalized = source_vector / np.linalg.norm(source_vector)
            #recovered_normalized = recovered / np.linalg.norm(recovered)
            
            assert recovered.shape == source_vector.shape, \
                f"{fwd_name}->{inv_name}: Shape mismatch {recovered.shape} != {source_vector.shape}"
            
            # Check if they are close (either directly or with a phase factor)
            try:
                np.testing.assert_allclose(source_vector, recovered, rtol=rtol, atol=atol)
                logger.debug(f"{fwd_name}->{inv_name}: PASSED (direct match)")
            except AssertionError:
                # Try with phase alignment
                phase_factor = np.vdot(recovered, source_vector) / np.vdot(source_vector, source_vector)
                phase_factor = phase_factor / abs(phase_factor)  # Normalize to unit magnitude
                try:
                    np.testing.assert_allclose(source_vector, recovered * phase_factor, rtol=rtol, atol=atol)
                    logger.debug(f"{fwd_name}->{inv_name}: PASSED (with phase correction)")
                except AssertionError:
                    logger.error(f"{fwd_name}->{inv_name}: FAILED - no good alignment found")
                    raise

# ============ EXISTING INDIVIDUAL TESTS (Keep these for backwards compatibility) ============

#def test_forward_inverse_by_hand(test_config_02, source_field_2x2):
#    logger.debug("==== Starting test_forward_inverse_by_hand ====")
#    config = test_config_02
#    source_points = create_points(config['source'])
#    receiving_points = create_points(config['receiver'])
#    k = 2 * np.pi
#    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
#    source = source_field_2x2.reshape(-1, 1)
#    source /= np.linalg.norm(source)  # Normalize source vector
#    received = forward_by_hand(Gsr, source)
#    recovered = inverse_by_hand(Gsr, received)
#    assert np.allclose(source, recovered, atol=1e-6)
#
#def test_forward_projection_direct_4x4(test_config_01, source_field_4x4):
#    logger.debug("==== Starting test_forward_projection_direct_4x4 ====")
#    config = test_config_01
#    source_points = create_points(config['source'])
#    receiving_points = create_points(config['receiver'])
#    k = 2 * np.pi
#    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
#    source_vector = source_field_4x4.reshape(-1, 1)
#    source_vector /= np.linalg.norm(source_vector)  # Normalize source vector
#    projected_field = forward_projection_direct(source_vector, Gsr)
#    expected_field = forward_by_hand(Gsr, source_field_4x4)
#    logger.debug(f"Projected field shape: {projected_field.shape}")
#    logger.debug(f"Expected field shape: {expected_field.shape}")
#    assert projected_field.shape == expected_field.shape
#    np.testing.assert_allclose(projected_field, expected_field, rtol=1e-5, atol=1e-8)
#
#def test_forward_projection_direct_2x2(test_config_02, source_field_2x2):
#    logger.debug("==== Starting test_forward_projection_direct_2x2 ====")
#    config = test_config_02
#    source_points = create_points(config['source'])
#    receiving_points = create_points(config['receiver'])
#    k = 2 * np.pi
#    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
#    source_vector = source_field_2x2.reshape(-1, 1)
#    source_vector /= np.linalg.norm(source_vector)  # Normalize source vector
#    projected_field = forward_projection_direct(source_vector, Gsr)
#    expected_field = forward_by_hand(Gsr, source_field_2x2)
#    logger.debug(f"Projected field shape: {projected_field.shape}")
#    logger.debug(f"Expected field shape: {expected_field.shape}")
#    assert projected_field.shape == expected_field.shape
#    np.testing.assert_allclose(projected_field, expected_field, rtol=1e-5, atol=1e-8)
#
#def test_forward_projection_eig_2x2(test_config_02, source_field_2x2):
#    logger.debug("==== Starting test_forward_projection_eig_2x2 ====")
#    config = test_config_02
#    source_points = create_points(config['source'])
#    receiving_points = create_points(config['receiver'])
#    k = 2 * np.pi
#    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
#    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
#    source_vector = source_field_2x2.reshape(-1, 1)
#    source_vector /= np.linalg.norm(source_vector)
#    projected_field = forward_projection_eig(source_vector, eig_vals, eig_vect_source, eig_vect_receiver)
#    expected_field = forward_by_hand(Gsr, source_field_2x2)
#    logger.debug(f"Projected field shape: {projected_field.shape}")
#    logger.debug(f"Expected field shape: {expected_field.shape}")
#    assert np.allclose(np.linalg.norm(source_vector), 1.0)
#    assert np.allclose(np.linalg.norm(eig_vect_source[:, 0]), 1.0)
#    assert np.allclose(np.linalg.norm(eig_vect_receiver[:, 0]), 1.0)
#    assert projected_field.shape == expected_field.shape, \
#        f"Shape mismatch: projected={projected_field.shape}, expected={expected_field.shape}"
#    np.testing.assert_allclose(projected_field, expected_field, rtol=1e-5, atol=1e-8)
#
#def test_forward_projection_eig_4x4(test_config_01, source_field_4x4):
#    logger.debug("==== Starting test_forward_projection_eig_4x4 ====")
#    config = test_config_01
#    source_points = create_points(config['source'])
#    receiving_points = create_points(config['receiver'])
#    k = 2 * np.pi
#    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
#    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
#    source_vector = source_field_4x4.reshape(-1, 1)
#    source_vector /= np.linalg.norm(source_vector)
#    projected_field = forward_projection_eig(source_vector, eig_vals, eig_vect_source, eig_vect_receiver)
#    expected_field = forward_by_hand(Gsr, source_field_4x4)
#    logger.debug(f"Projected field shape: {projected_field.shape}")
#    logger.debug(f"Expected field shape: {expected_field.shape}")
#    assert np.allclose(np.linalg.norm(source_vector), 1.0)
#    assert np.allclose(np.linalg.norm(eig_vect_source[:, 0]), 1.0)
#    assert np.allclose(np.linalg.norm(eig_vect_receiver[:, 0]), 1.0)
#    assert projected_field.shape == expected_field.shape, \
#        f"Shape mismatch: projected={projected_field.shape}, expected={expected_field.shape}"
#    np.testing.assert_allclose(projected_field, expected_field, rtol=1e-5, atol=1e-8)
#
#def test_inverse_projection_direct_4x4(test_config_01, source_field_4x4):
#    logger.debug("==== Starting test_inverse_projection_direct_4x4 ====")
#    config = test_config_01
#    source_points = create_points(config['source'])
#    receiving_points = create_points(config['receiver'])
#    k = 2 * np.pi
#    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
#    source_vector = source_field_4x4.reshape(-1, 1)
#    source_vector /= np.linalg.norm(source_vector)
#    # Project the source field to get the received field
#    projected_field = forward_by_hand(Gsr, source_field_4x4)
#    # Now test the inverse projection
#    expected_field = inverse_by_hand(Gsr, projected_field)
#    inverse_projected_field = inverse_projection_direct(projected_field, Gsr)
#    logger.debug(f"Inverse projected field shape: {inverse_projected_field.shape}")
#    # Check that the expected field matches the original source field
#    assert inverse_projected_field.shape == expected_field.shape
#    assert inverse_projected_field.shape == source_vector.shape
#    np.testing.assert_allclose(inverse_projected_field, source_vector, rtol=1e-5, atol=1e-8)
#    np.testing.assert_allclose(inverse_projected_field, expected_field, rtol=1e-5, atol=1e-8)
#
#def test_inverse_projection_direct_2x2(test_config_02, source_field_2x2):
#    logger.debug("==== Starting test_inverse_projection_direct_2x2 ====")
#    config = test_config_02
#    source_points = create_points(config['source'])
#    receiving_points = create_points(config['receiver'])
#    k = 2 * np.pi
#    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
#    source_vector = source_field_2x2.reshape(-1, 1)
#    source_vector /= np.linalg.norm(source_vector)
#    # Project the source field to get the received field
#    projected_field = forward_by_hand(Gsr, source_field_2x2)
#    # Now test the inverse projection
#    expected_field = inverse_by_hand(Gsr, projected_field)
#    inverse_projected_field = inverse_projection_direct(projected_field, Gsr)
#    logger.debug(f"Inverse projected field shape: {inverse_projected_field.shape}")
#    # Check that the expected field matches the original source field
#    assert inverse_projected_field.shape == expected_field.shape
#    assert inverse_projected_field.shape == source_vector.shape
#    np.testing.assert_allclose(inverse_projected_field, source_vector, rtol=1e-5, atol=1e-8)
#    np.testing.assert_allclose(inverse_projected_field, expected_field, rtol=1e-5, atol=1e-8)
#
#@pytest.mark.parametrize("rtol", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
#@pytest.mark.parametrize("atol", [1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
#def test_inverse_projection_eig_4x4(test_config_01, source_field_4x4, rtol, atol):
#    logger.debug("==== Starting test_inverse_projection_eig_4x4 ====")
#    config = test_config_01
#    source_points = create_points(config['source'])
#    receiving_points = create_points(config['receiver'])
#    k = 2 * np.pi
#    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
#    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
#    source_vector = source_field_4x4.reshape(-1, 1)
#    source_vector /= np.linalg.norm(source_vector)
#    # Project the source field to get the received field
#    projected_field = forward_projection_eig(source_vector, eig_vals, eig_vect_source, eig_vect_receiver)
#    # Now test the inverse projection
#    expected_field = inverse_projection_eig(projected_field, eig_vals, eig_vect_source, eig_vect_receiver)
#    inverse_projected_field = inverse_projection_direct(projected_field, Gsr)
#    logger.debug(f"Inverse projected field shape: {inverse_projected_field.shape}")
#    # Check that the expected field matches the original source field
#    assert inverse_projected_field.shape == expected_field.shape
#    assert inverse_projected_field.shape == source_vector.shape
#    np.testing.assert_allclose(inverse_projected_field, source_vector, rtol=rtol, atol=atol)
#    np.testing.assert_allclose(inverse_projected_field, expected_field, rtol=rtol, atol=atol)
#
#@pytest.mark.parametrize("rtol", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
#@pytest.mark.parametrize("atol", [1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
#def test_inverse_projection_eig_2x2(test_config_02, source_field_2x2, rtol, atol):
#    logger.debug("==== Starting test_inverse_projection_eig_2x2 ====")
#    config = test_config_02
#    source_points = create_points(config['source'])
#    receiving_points = create_points(config['receiver'])
#    k = 2 * np.pi
#    Gsr = free_space_transfer_function(source_points, receiving_points, k, method='direct')
#    eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)
#    source_vector = source_field_2x2.reshape(-1, 1)
#    source_vector /= np.linalg.norm(source_vector)
#    # Project the source field to get the received field
#    projected_field = forward_projection_eig(source_vector, eig_vals, eig_vect_source, eig_vect_receiver)
#    # Now test the inverse projection
#    expected_field = inverse_by_hand(Gsr, projected_field)
#    inverse_projected_field = inverse_projection_direct(projected_field, Gsr)
#    logger.debug(f"Inverse projected field shape: {inverse_projected_field.shape}")
#    # Check that the expected field matches the original source field
#    assert inverse_projected_field.shape == expected_field.shape
#    assert inverse_projected_field.shape == source_vector.shape
#    np.testing.assert_allclose(inverse_projected_field, source_vector, rtol=rtol, atol=atol)
#    np.testing.assert_allclose(inverse_projected_field, expected_field, rtol=rtol, atol=atol)
#
