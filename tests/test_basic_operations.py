import numpy as np
import pytest




#===================================================
# Testing numpy transpose functionality
#===================================================
@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (4, 4), (5, 5), (100, 100), (324, 256)])
def test_numpy_transpose(shape):
    """
    Test numpy transpose functionality for various shapes.
    
    Args:
        shape (tuple): Shape of the array to be tested.
    """
    arr = np.random.rand(*shape)
    transposed = np.transpose(arr)
    assert transposed.shape == (shape[1], shape[0]), \
        f"Transposed shape {transposed.shape} does not match expected {shape[1], shape[0]}"
    assert np.allclose(arr.T, transposed), "Transposed array does not match numpy's transpose"

#===================================================
# Testing numpy transpose with complex numbers
#===================================================
@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (4, 4), (5, 5), (100, 100), (324, 256)])
def test_numpy_transpose_complex(shape):
    """
    Test numpy transpose functionality with complex numbers for various shapes.
    
    Args:
        shape (tuple): Shape of the array to be tested.
    """
    arr = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    transposed = np.transpose(arr)
    assert transposed.shape == (shape[1], shape[0]), \
        f"Transposed shape {transposed.shape} does not match expected {shape[1], shape[0]}"
    assert np.allclose(arr.T, transposed), "Transposed complex array does not match numpy's transpose"

#===================================================
# Testing numpy complex conjugate functionality
#===================================================
@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (4, 4), (5, 5), (100, 100), (324, 256)])
def test_numpy_conjugate(shape):
    """
    Test numpy complex conjugate functionality for various shapes.
    
    Args:
        shape (tuple): Shape of the array to be tested.
    """
    arr = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    conjugated = np.conj(arr)
    assert conjugated.shape == shape, \
        f"Conjugated shape {conjugated.shape} does not match expected {shape}"
    assert np.allclose(np.conj(arr), conjugated), "Conjugated array does not match numpy's conjugate"

#===================================================
# Testing numpy complex conjugate with transpose
#===================================================
def test_numpy_conjugate_transpose():
    data = [ [1 + 2j, 3 + 4j], 
             [5 + 6j, 7 + 8j] ]
    expected = [ [1 - 2j, 5 - 6j], 
                 [3 - 4j, 7 - 8j] ]
    arr = np.array(data)
    conjugated_transposed = np.conj(arr.T)
    assert np.array_equal(conjugated_transposed, expected), \
        "Conjugated transposed array does not match expected result"

#=====================================================
# Testing conjugate transpose vs transpose conjugate
#=====================================================
def test_conjugate_transpose_vs_transpose_conjugate():
    """
    Test that the conjugate transpose is equal to the transpose of the conjugate.
    """
    arr = np.array([[1 + 2j, 3 + 4j], 
                    [5 + 6j, 7 + 8j]])
    conjugate_transpose = np.conj(arr.T)
    transpose_conjugate = (np.conj(arr)).T
    assert np.array_equal(conjugate_transpose, transpose_conjugate), \
        "Conjugate transpose does not match transpose of conjugate"

#===================================================
# Testing @ operator for matrix multiplication, hand calculated value comparison
#===================================================
def test_matrix_multiplication():
    """
    Test @ operator for matrix multiplication with hand-calculated values.
    """
    A = np.array([[1, 2], 
                  [3, 4]])
    B = np.array([[5, 6], 
                  [7, 8]])
    # (1 * 5 + 2 * 7 = 19, 1 * 6 + 2 * 8 = 22)
    # (3 * 5 + 4 * 7 = 43, 3 * 6 + 4 * 8 = 50)
    expected = np.array([[19, 22], [43, 50]])
    
    result = A @ B
    assert np.allclose(result, expected), \
        f"Matrix multiplication result {result} does not match expected {expected}"

#===================================================
# Testing @ operator for matrix multiplication with complex numbers, hand calculated value comparison
#===================================================
def test_matrix_multiplication_complex():
    """
    Test @ operator for matrix multiplication with complex numbers and hand-calculated values.
    """
    A = np.array([[1 + 2j, 3 + 4j], 
                  [5 + 6j, 7 + 8j]])
    B = np.array([[9 + 10j, 11 + 12j], 
                  [13 + 14j, 15 + 16j]])

    # (1 + 2j) * (9 + 10j) + (3 + 4j) * (13 + 14j)
    # = (9 + 10j + 18j - 20) + (39 + 42j + 52j - 56)
    # = (-11 + 28j) + (-17 + 94j) = (-28 + 122j)
    expected_00 = (-28 + 122j)

    # (1 + 2j) * (11 + 12j) + (3 + 4j) * (15 + 16j)
    # = (11 + 12j + 22j - 24) + (45 + 48j + 60j - 64)
    # = (-13 + 34j) + (-19 + 108j) = (-32 + 142j)
    expected_01 = (-32 + 142j)

    # (5 + 6j) * (9 + 10j) + (7 + 8j) * (13 + 14j)
    # = (45 + 50j + 54j - 60) + (91 + 98j + 104j - 112)
    # = (-15 + 104j) + (-21 + 202j) = (-36 + 306j)
    expected_10 = (-36 + 306j)

    # (5 + 6j) * (11 + 12j) + (7 + 8j) * (15 + 16j)
    # = (55 + 60j + 66j - 72) + (105 + 112j + 120j - 128)
    # = (-17 + 126j) + (-23 + 232j) = (-40 + 358j)
    expected_11 = (-40 + 358j)

    expected = np.array([[expected_00, expected_01],
                         [expected_10, expected_11]])
    
    result = A @ B
    assert np.allclose(result, expected), \
        f"Complex matrix multiplication result {result} does not match expected {expected}"

#===================================================
# Testing numpy matrix multiplication
#===================================================
def test_matrix_multiplication_numpy():
    A = np.array([[1, 2], 
                  [3, 4]])
    B = np.array([[5, 6], 
                  [7, 8]])
    # (1 * 5 + 2 * 7 = 19, 1 * 6 + 2 * 8 = 22)
    # (3 * 5 + 4 * 7 = 43, 3 * 6 + 4 * 8 = 50)
    expected = np.array([[19, 22], [43, 50]])
    result = np.matmul(A, B)
    assert np.allclose(result, expected), \
        f"Numpy matrix multiplication result {result} does not match expected {expected}"


#===================================================
# Testing numpy matrix multiplication with complex numbers
#===================================================
def test_matrix_multiplication_numpy_complex():
    A = np.array([[1 + 2j, 3 + 4j], 
                  [5 + 6j, 7 + 8j]])
    B = np.array([[9 + 10j, 11 + 12j], 
                  [13 + 14j, 15 + 16j]])

    # Same as above
    # (1 + 2j) * (9 + 10j) + (3 + 4j) * (13 + 14j)
    expected_00 = (-28 + 122j)

    # (1 + 2j) * (11 + 12j) + (3 + 4j) * (15 + 16j)
    expected_01 = (-32 + 142j)

    # (5 + 6j) * (9 + 10j) + (7 + 8j) * (13 + 14j)
    expected_10 = (-36 + 306j)

    # (5 + 6j) * (11 + 12j) + (7 + 8j) * (15 + 16j)
    expected_11 = (-40 + 358j)

    expected = np.array([[expected_00, expected_01], 
                         [expected_10, expected_11]])
    
    result = np.matmul(A, B)
    assert np.allclose(result, expected), \
        f"Numpy complex matrix multiplication result {result} does not match expected {expected}"


