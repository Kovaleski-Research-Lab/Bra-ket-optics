import numpy as np
import pytest

from src.eig import (
    direct_transfer_function,
    blockwise_transfer_function,
    free_space_transfer_function
)

#===============================================
# Test cases for implementation against hand calculated values
#===============================================
def test_direct_transfer_function_hand_calculated():
    # Hand-calculated values for a simple case
    s = np.array([[1, 0, 0]]) #x,y,z
    r = np.array([[0, 2, 3]]) #x,y,z
    k = 2 * np.pi  # Wavenumber for λ = 1

    distance = np.sqrt((s[0, 0] - r[0, 0])**2 + (s[0, 1] - r[0, 1])**2 + (s[0, 2] - r[0, 2])**2)
    # Expected value for this configuration
    numerator = -np.exp(1j * k * distance)
    denominator = 4 * np.pi * distance
    expected_value = numerator / denominator
    g = direct_transfer_function(s, r, k)

    assert g.shape == (1, 1)
    assert np.isclose(g[0, 0], expected_value, rtol=1e-10, atol=1e-12), \
        f"Expected {expected_value}, got {g[0, 0]}"

def test_blockwise_transfer_function_hand_calculated():
    # Hand-calculated values for a simple case
    s = np.array([[1, 0, 0]]) #x,y,z
    r = np.array([[0, 2, 3]]) #x,y,z
    k = 2 * np.pi  # Wavenumber for λ = 1

    distance = np.sqrt((s[0, 0] - r[0, 0])**2 + (s[0, 1] - r[0, 1])**2 + (s[0, 2] - r[0, 2])**2)
    # Expected value for this configuration
    numerator = -np.exp(1j * k * distance)
    denominator = 4 * np.pi * distance
    expected_value = numerator / denominator
    g = blockwise_transfer_function(s, r, k, blocksize=1)

    assert g.shape == (1, 1)
    assert np.isclose(g[0, 0], expected_value, rtol=1e-10, atol=1e-12), \
        f"Expected {expected_value}, got {g[0, 0]}" 

def test_direct_transfer_function_hand_calculated_multi():
    # Hand-calculated values for a simple case with multiple sources and receivers
    s = np.array([[1, 0, 0], [0, 1, 0]]) # Two source points
    r = np.array([[0, 2, 3], [3, 0, 4]]) # Two receiver points
    k = 2 * np.pi  # Wavenumber for λ = 1

    g = direct_transfer_function(s, r, k)

    assert g.shape == (len(r), len(s))
    assert g.dtype == np.complex128

    # Check individual values against hand calculations
    for i in range(len(r)):
        for j in range(len(s)):
            distance = np.sqrt((s[j, 0] - r[i, 0])**2 + (s[j, 1] - r[i, 1])**2 + (s[j, 2] - r[i, 2])**2)
            numerator = -np.exp(1j * k * distance)
            denominator = 4 * np.pi * distance
            expected_value = numerator / denominator
            assert np.isclose(g[i, j], expected_value, rtol=1e-10, atol=1e-12), \
                f"Expected {expected_value}, got {g[i, j]}"

def test_blockwise_transfer_function_hand_calculated_multi():
    # Hand-calculated values for a simple case with multiple sources and receivers
    s = np.array([[1, 0, 0], [0, 1, 0]]) # Two source points
    r = np.array([[0, 2, 3], [3, 0, 4]]) # Two receiver points
    k = 2 * np.pi  # Wavenumber for λ = 1

    g = blockwise_transfer_function(s, r, k, blocksize=1)

    assert g.shape == (len(r), len(s))
    assert g.dtype == np.complex128

    # Check individual values against hand calculations
    for i in range(len(r)):
        for j in range(len(s)):
            distance = np.sqrt((s[j, 0] - r[i, 0])**2 + (s[j, 1] - r[i, 1])**2 + (s[j, 2] - r[i, 2])**2)
            numerator = -np.exp(1j * k * distance)
            denominator = 4 * np.pi * distance
            expected_value = numerator / denominator
            assert np.isclose(g[i, j], expected_value, rtol=1e-10, atol=1e-12), \
                f"Expected {expected_value}, got {g[i, j]}"

#===============================================
# Test cases for transfer function calculations
#===============================================
@pytest.fixture
def test_data():
    # Set random seed for reproducibility
    np.random.seed(42)

    Ns, Nr = 100, 120
    s = np.random.rand(Ns, 3)  # Source points (Ns, 3)
    r = np.random.rand(Nr, 3)  # Receiver points (Nr, 3)
    k = 2 * np.pi  # Wavenumber (e.g., for λ = 1)

    return s, r, k

def test_direct_vs_blockwise_transfer_function(test_data):
    s, r, k = test_data

    g_direct = direct_transfer_function(s, r, k)
    g_blockwise = blockwise_transfer_function(s, r, k, blocksize=4)

    assert g_direct.shape == g_blockwise.shape
    assert np.allclose(g_direct, g_blockwise, rtol=1e-10, atol=1e-12), \
            f"Max difference: {np.max(np.abs(g_direct - g_blockwise))}"

def test_free_space_transfer_function_methods_equivalence(test_data):
    s, r, k = test_data

    g_direct = free_space_transfer_function(s, r, k, method='direct')
    g_blockwise = free_space_transfer_function(s, r, k, method='blockwise', blocksize=4)

    assert g_direct.shape == g_blockwise.shape
    assert np.allclose(g_direct, g_blockwise, rtol=1e-10, atol=1e-12), \
        f"Max difference: {np.max(np.abs(g_direct - g_blockwise))}"

def test_invalid_method_raises(test_data):
    s, r, k = test_data

    with pytest.raises(ValueError, match="Method not recognized"):
        _ = free_space_transfer_function(s, r, k, method='invalid')

#===============================================
# Test cases for checking dtypes and shapes
#===============================================
def test_transfer_function_dtype_and_shape(test_data):
    s, r, k = test_data

    g_direct = direct_transfer_function(s, r, k)
    g_blockwise = blockwise_transfer_function(s, r, k, blocksize=4)

    assert g_direct.dtype == np.complex128
    assert g_blockwise.dtype == np.complex128

    assert g_direct.shape == (len(r), len(s))
    assert g_blockwise.shape == (len(r), len(s))

    # Check if the shapes are consistent
    assert g_direct.shape == g_blockwise.shape

def test_free_space_function_dtype_and_shape(test_data):
    s, r, k = test_data

    g_direct = free_space_transfer_function(s, r, k, method='direct')
    g_blockwise = free_space_transfer_function(s, r, k, method='blockwise', blocksize=4)

    assert g_direct.dtype == np.complex128
    assert g_blockwise.dtype == np.complex128

    assert g_direct.shape == (len(r), len(s))
    assert g_blockwise.shape == (len(r), len(s))

    # Check if the shapes are consistent
    assert g_direct.shape == g_blockwise.shape

#===============================================
# Test cases for transfer functions across scales
#===============================================
@pytest.mark.parametrize("scale", [1e-150, 1e-100, 1e-50, 1.0, 1e50, 1e100, 1e150])
def test_transfer_functions_consistent_across_scales(scale):
    np.random.seed(42)
    Ns, Nr = 100, 120
    s = scale * np.random.rand(Ns, 3)
    r = scale * np.random.rand(Nr, 3)
    k = 2 * np.pi / scale  # Adjust k so wavelength fits the scale

    g_direct = direct_transfer_function(s, r, k)
    g_blockwise = blockwise_transfer_function(s, r, k, blocksize=4)

    assert g_direct.shape == g_blockwise.shape
    assert np.allclose(g_direct, g_blockwise, rtol=1e-10, atol=1e-12), \
        f"Max difference: {np.max(np.abs(g_direct - g_blockwise))}"

@pytest.mark.parametrize("scale", [1e-150, 1e-100, 1e-50, 1.0, 1e50, 1e100, 1e150])
def test_free_space_function_consistency(scale):
    np.random.seed(42)
    Ns, Nr = 100, 120
    s = scale * np.random.rand(Ns, 3)
    r = scale * np.random.rand(Nr, 3)
    k = 2 * np.pi / scale

    g_direct = free_space_transfer_function(s, r, k, method='direct')
    g_blockwise = free_space_transfer_function(s, r, k, method='blockwise', blocksize=4)

    assert g_direct.shape == g_blockwise.shape
    assert np.allclose(g_direct, g_blockwise, rtol=1e-10, atol=1e-12), \
        f"Max difference: {np.max(np.abs(g_direct - g_blockwise))}"

# ===============================================
# Testing with using geometry imports
# ===============================================
from src.geometry import create_points
from tests.generate_test_configs import generate_configs

@pytest.mark.parametrize("config", generate_configs(100, 1337))
def test_transfer_function_with_geometry(config):
    s = create_points(config['source'])
    r = create_points(config['receiver'])
    k = 2 * np.pi  # Wavenumber for λ = 1

    g_direct = direct_transfer_function(s, r, k)
    g_blockwise = blockwise_transfer_function(s, r, k, blocksize=20)

    assert g_direct.shape == g_blockwise.shape
    assert np.allclose(g_direct, g_blockwise, rtol=1e-10, atol=1e-12), \
        f"Max difference: {np.max(np.abs(g_direct - g_blockwise))}"

@pytest.mark.parametrize("config", generate_configs(100, 1337))
def test_free_space_function_with_geometry(config):
    s = create_points(config['source'])
    r = create_points(config['receiver'])
    k = 2 * np.pi  # Wavenumber for λ = 1

    g_direct = free_space_transfer_function(s, r, k, method='direct')
    g_blockwise = free_space_transfer_function(s, r, k, method='blockwise', blocksize=20)

    assert g_direct.shape == g_blockwise.shape
    assert np.allclose(g_direct, g_blockwise, rtol=1e-10, atol=1e-12), \
        f"Max difference: {np.max(np.abs(g_direct - g_blockwise))}"

