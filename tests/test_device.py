import pytest
import torch
from src.device import Device
from src.antenna import Antenna

@pytest.mark.order(2)
# Test: Initialize a rectangular device and verify internal state
def test_device_initialization_rectangular():
    device = Device(
        type="transmitter",
        center=[0.0, 0.0, 0.0],
        shape="rect",
        extent=[2.0, 2.0],
        discretization=10
    )

    # Check basic attributes and shape of generated grid
    assert device.shape == "rect"
    assert device.extent.tolist() == [2.0, 2.0]
    assert device.radius is None
    assert device.X.shape == (10, 10)
    assert device.Y.shape == (10, 10)
    assert device.grid.shape == (2, 10, 10)


# Test: Initializing a device without extent or radius should raise an error
def test_device_initialization_fails_on_missing_geometry():
    with pytest.raises(ValueError, match="Either extent or radius must be provided"):
        _ = Device(
            type="receiver",
            center=[0.0, 0.0, 0.0],
            shape="rect"
        )


# Test: Initializing a device with both extent and radius should raise an error
def test_device_initialization_fails_on_both_extent_and_radius():
    with pytest.raises(ValueError, match="Only one of extent or radius should be provided"):
        _ = Device(
            type="receiver",
            center=[0.0, 0.0, 0.0],
            shape="rect",
            extent=[1.0, 1.0],
            radius=1.0
        )


# Test: Add an antenna that is well within the device bounds — should succeed
def test_add_antenna_within_bounds():
    device = Device(
        type="transmitter",
        center=[0.0, 0.0, 0.0],
        shape="rect",
        extent=[2.0, 2.0]
    )
    antenna = Antenna(center=[0.0, 0.0, 0.0], wavelength=1.0)
    device.add_antenna(antenna)

    # Check that antenna was added and its position is stored
    assert len(device.antennas) == 1
    assert device.antennas[0].get_position().tolist() == [0.0, 0.0, 0.0]


# Test: Add an antenna that is outside the bounds of a rectangular device — should raise an error
def test_add_antenna_out_of_bounds():
    device = Device(
        type="receiver",
        center=[0.0, 0.0, 0.0],
        shape="rect",
        extent=[1.0, 1.0]
    )
    antenna = Antenna(center=[10.0, 10.0, 0.0], wavelength=1.0)

    # Antenna position is far outside device bounds
    with pytest.raises(ValueError, match="Antenna at position"):
        device.add_antenna(antenna)


# Test: Try to create a device with an unsupported shape — should raise an error
def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="Invalid shape"):
        _ = Device(
            type="transmitter",
            center=[0.0, 0.0, 0.0],
            shape="circle",
            radius=1.0
        )

