import torch
from device import Device

class Domain:
    def __init__(self, config: dict):
        self.config = config

    def generate_grid(self):
        pass

    def add_device(self, device: Device):
        pass
    
