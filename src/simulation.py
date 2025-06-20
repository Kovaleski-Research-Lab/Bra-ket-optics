import torch

from utils import euclidean_distance, sum_rule


class Simulation(config: dict):
    def __init__(self, config: dict):
        self.config = config
        self.wavelength = config["wavelength"]
        self.Ns = config["Ns"]
        self.Nr = config["Nr"]
        self.Lxs = config["Lxs"]
        self.Lys = config["Lys"]
        self.Lxr = config["Lxr"]
        self.Lyr = config["Lyr"]

    def _generate_sources(self):
        # Generates the source points in (x, y) coordinates.
        self.sources = torch.zeros((self.Ns, 2), dtype=torch.float64)
        self.sources[:, 0] = torch.linspace(0, self.Lxs, self.Ns)
        self.sources[:, 1] = torch.linspace(0, self.Lys, self.Ns)
        return self.sources
    
    def _generate_receivers(self):
        pass
    
    def _generate_transfer_function(self):

    def run(self):
        pass

    def plot(self):
        pass


if __name__ == "__main__":
    import yaml
    config = yaml.load(open("../config.yaml"))
    sim = Simulation(config)