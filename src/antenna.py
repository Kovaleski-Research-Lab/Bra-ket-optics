import torch


class Antenna(config: dict):
    def __init__(self, config: dict):
        self.config = config

        self.center = config["center"]
        self.type = config["type"]
        self.wavelength = config["wavelength"]
        