import torch
from typing import Optional

class Device:
    def __init__(self, type: str, 
                 center: list[float], 
                 shape: str, 
                 extent: Optional[list[float]] = None,
                 radius: Optional[float] = None):
        self.type = type
        self.center = center
        self.shape = shape
        self.extent = extent
        self.radius = radius
        self.antennas = []

    def build_device(self):
        if self.shape == "rect":
            self.build_rect_device()
        elif self.shape == "circle":
            self.build_circle_device()
        else:
            raise ValueError(f"Invalid shape: {self.shape}")
        
    def build_rect_device(self):
        if self.extent is None:
            raise ValueError("Extent is required for rectangular devices")
        if self.radius is not None:
            raise ValueError("Radius is not allowed for rectangular devices")
        self.Lx = self.extent[0]
        self.Ly = self.extent[1]
        self.x = torch.linspace(self.center[0] - self.Lx/2, self.center[0] + self.Lx/2, self.Lx)
        self.y = torch.linspace(self.center[1] - self.Ly/2, self.center[1] + self.Ly/2, self.Ly)
        self.X, self.Y = torch.meshgrid(self.x, self.y)
        self.grid = torch.stack([self.X, self.Y], dim=0)

    def build_circle_device(self):
        if self.radius is None:
            raise ValueError("Radius is required for circular devices")
        if self.extent is not None:
            raise ValueError("Extent is not allowed for circular devices")
        self.radius = torch.tensor(self.radius)
        self.x = torch.linspace(self.center[0] - self.radius, self.center[0] + self.radius, self.radius)
        self.y = torch.linspace(self.center[1] - self.radius, self.center[1] + self.radius, self.radius)
        self.X, self.Y = torch.meshgrid(self.x, self.y)
        self.grid = torch.stack([self.X, self.Y], dim=0)

    def add_antenna(self, antenna: dict):
        self.antennas.append(antenna)
        self._check_antenna_extents()

    def _check_antenna_extents(self):
        # Checks that the centers of the antennas are within the extent of the device
        pass