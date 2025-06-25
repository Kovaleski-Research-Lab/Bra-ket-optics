import matplotlib.pyplot as plt
import numpy as np

from device import Device


def plot_device(device: Device):
    plt.figure()
    plt.scatter(device.grid[0], device.grid[1])
    plt.show()


if __name__ == "__main__":
    device = Device(type="rect", center=[0, 0], shape="rect", extent=[1, 1])
    device.build_device()
    plot_device(device)