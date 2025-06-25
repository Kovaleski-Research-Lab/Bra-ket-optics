import torch
import pytest
from src.antenna import Antenna

@pytest.mark.order(1)

# Test: Check initialization for different gradient settings and confirm gradient flags
@pytest.mark.parametrize("gradients,amp_grad,phase_grad", [
    ("none", False, False),
    ("complex", True, True),
    ("amplitude_only", True, False),
    ("phase_only", False, True),
])
def test_initialization(gradients, amp_grad, phase_grad):
    antenna = Antenna(center=[1.0, 2.0, 3.0], wavelength=0.5, gradients=gradients)

    # Validate position tensor
    assert torch.allclose(antenna.get_position(), torch.tensor([1.0, 2.0, 3.0]))
    
    # Confirm gradient requirements match expected behavior
    assert antenna.get_amplitude().requires_grad == amp_grad
    assert antenna.get_phase().requires_grad == phase_grad
    
    # Ensure transmissivity is a complex number
    assert torch.is_complex(antenna.get_transmissivity_function())


# Test: Set amplitude and confirm both value and updated transmissivity
def test_set_amplitude():
    antenna = Antenna(center=[0, 0, 0], wavelength=1.0)
    antenna.set_amplitude(3.5)

    # Validate amplitude is updated
    assert torch.allclose(antenna.get_amplitude(), torch.tensor([3.5]))

    # Confirm transmissivity function is recalculated correctly
    expected_tf = 3.5 * torch.exp(1j * antenna.get_phase())
    assert torch.allclose(antenna.get_transmissivity_function(), expected_tf)


# Test: Set phase and validate it updates correctly and affects transmissivity
def test_set_phase():
    antenna = Antenna(center=[0, 0, 0], wavelength=1.0)
    antenna.set_phase(torch.pi / 2)

    # Confirm phase is set
    assert torch.allclose(antenna.get_phase(), torch.tensor([torch.pi / 2]))

    # Validate transmissivity function updates correctly
    expected_tf = antenna.get_amplitude() * torch.exp(1j * torch.tensor([torch.pi / 2]))
    assert torch.allclose(antenna.get_transmissivity_function(), expected_tf)


# Test: Set full transmissivity as a complex number and ensure amplitude and phase decompose properly
def test_set_transmissivity_function():
    antenna = Antenna(center=[0, 0, 0], wavelength=1.0)
    tf = 2 * torch.exp(1j * torch.tensor(0.25 * torch.pi))
    antenna.set_transmissivity_function(tf)

    # Check decomposition into amplitude and phase
    assert torch.allclose(antenna.get_amplitude(), torch.tensor([2.0]), atol=1e-4)
    assert torch.allclose(antenna.get_phase(), torch.tensor([0.25 * torch.pi]), atol=1e-4)

    # Verify recomposed transmissivity function matches input
    expected_tf = 2.0 * torch.exp(1j * torch.tensor([0.25 * torch.pi]))
    assert torch.allclose(antenna.get_transmissivity_function(), expected_tf, atol=1e-4)


# Test: Check that the forward pass multiplies input tensor by the transmissivity function
def test_forward():
    antenna = Antenna(center=[0, 0, 0], wavelength=1.0)
    input_tensor = torch.tensor([1.0 + 1.0j])
    output = antenna(input_tensor)

    # Forward output should be transmissivity * input
    expected = antenna.get_transmissivity_function() * input_tensor
    assert torch.allclose(output, expected)

