import torch
from typing import Optional
from antenna import Antenna

class Device:
    """
    A class representing a device in a simulation environment. 
    
    Attributes:
        type (str): Type of the device, e.g., 'transmitter' or 'receiver'.
        center (list[float]): Center coordinates of the device [x, y, z].
        shape (str): Shape of the device, 'rect' is currently supported.
        extent (Optional[list[float]]): Extent of the device [Lx, Ly] for rectangular devices.
        radius (Optional[float]): Radius for circular devices.
        antennas (list): List of antennas associated with the device.
    """
    def __init__(self, type: str, 
                 center: list[float], 
                 shape: str, 
                 discretization: int = 100,
                 extent: Optional[list[float]] = None,
                 radius: Optional[float] = None):
        self.type = type
        self.center = torch.tensor(center)
        self.shape = shape
        self.discretization = discretization

        if extent is None and radius is None:
            raise ValueError("Either extent or radius must be provided")
        if extent is not None and radius is not None:
            raise ValueError("Only one of extent or radius should be provided")

        if extent is not None:
            if not torch.is_tensor(extent):
                self.extent = torch.tensor(extent)
            else:
                self.extent = extent
        else:
            self.extent = None

        if radius is not None:
            if not torch.is_tensor(radius):
                self.radius = torch.tensor(radius)
            else:
                self.radius = radius
        else:
            self.radius = None

        self.antennas = []
        self.build_device()

    def build_device(self):
        if self.shape == "rect":
            self.build_rect_device()
        else:
            raise ValueError(f"Invalid shape: {self.shape}")
        
    def build_rect_device(self):
        if self.extent is None:
            raise ValueError("Extent is required for rectangular devices")
        if self.radius is not None:
            raise ValueError("Radius is not allowed for rectangular devices")
        self.Lx = self.extent[0]
        self.Ly = self.extent[1]
        self.x = torch.linspace(self.center[0] - self.Lx/2, self.center[0] + self.Lx/2, self.discretization)
        self.y = torch.linspace(self.center[1] - self.Ly/2, self.center[1] + self.Ly/2, self.discretization)
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='ij')
        self.grid = torch.stack([self.X, self.Y], dim=0)

    def add_antenna(self, antenna: Antenna):
        self.antennas.append(antenna)
        self._check_antenna_extents()

    def _check_antenna_extents(self):
        for antenna in self.antennas:
            pos = antenna.get_position()
            if self.shape == "rect":
                if self.extent is None:
                    raise ValueError("Extent is required for rectangular devices")
                if not (self.center[0] - self.extent[0]/2 <= pos[0] <= self.center[0] + self.extent[0]/2 and
                        self.center[1] - self.extent[1]/2 <= pos[1] <= self.center[1] + self.extent[1]/2):
                    raise ValueError(f"Antenna at position {pos} is out of device bounds")
            # Add checks here for other shapes if needed
            else:
                raise ValueError(f"Invalid shape: {self.shape}")
        pass


if __name__ == "__main__":

    typ = "transmitter" 
    center = [0.0, 0.0, 0.0]
    shape = "rect"
    extent = [10., 10., 0.]
    device = Device(type=typ, center=center, shape=shape, extent=extent)
    print("Device Type:", device.type)
    print("Device Center:", device.center)
    print("Device Shape:", device.shape)
    print("Device Extent:", device.extent)
    print("Device Grid X:", device.X)
    print("Device Grid Y:", device.Y)
    print("Device Grid:", device.grid)

    antenna_center = [1.0, 1.0, 0.0]
    wavelength = 0.3
    antenna = Antenna(center=antenna_center, wavelength=wavelength, gradients='complex')
    device.add_antenna(antenna)
    print("Number of Antennas in Device:", len(device.antennas))
    print("Antenna Center:", device.antennas[0].get_position())


