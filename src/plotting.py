import numpy as np
import matplotlib.pyplot as plt


def plot_modes(xx, yy, zz, modes, plane='yz', x=None, y=None, z=None):
    """
    Plots the real part of the optical modes in a specified slice plane.

    Args:
        xx, yy, zz (np.ndarray): Meshgrid coordinate arrays.
        modes (np.ndarray): Modes evaluated at meshgrid points (Nmodes, ...).
        plane (str): Plane to plot ('xy', 'yz', or 'xz').
        x, y, z (int): Slice index to use for the specified plane.
    """
    print("Plotting modes...")
    modes = np.asarray([np.reshape(i, zz.shape) for i in modes])

    if plane == 'yz':
        if x is None:
            raise ValueError("For 'yz' plane, x-coordinate must be specified.")
        for mode in modes:
            sliced_modes = mode[x, :, :].squeeze()
            fig, ax = plt.subplots(figsize=(5,5))
            ax.imshow(np.real(sliced_modes), cmap='jet')
            plt.show()

    elif plane == 'xz':
        sliced_modes = modes[:, :, 0, :].squeeze()
        # Optional: Add plot here if needed

    elif plane == 'xy':
        if z is None:
            raise ValueError("For 'xy' plane, z-coordinate must be specified.")
        for i, mode in enumerate(modes):
            sliced_modes = mode[:, :, z].squeeze()
            fig, ax = plt.subplots(figsize=(5,5))
            fig.suptitle(f"Mode {i+1}")
            ax.imshow(np.real(sliced_modes), cmap='jet')
            plt.show()
    else:
        raise ValueError("Plane must be 'yz', 'xz', or 'xy'.")


def plot_mode_strengths(eig_vals, S):
    """
    Plots the mode strengths, and mode strengths normalized by total energy.
    Args:
        eig_vals (np.ndarray): Eigenvalues representing mode strengths.
        S (float): Total energy or normalization factor.
    """
    print("Plotting mode strengths...")
    # Plot mode strengths
    fig, ax = plt.subplots(2,2,figsize=(12, 10))
    ax[0][0].plot(eig_vals, marker='o', color='purple')
    ax[0][0].set_xlabel("Mode Number")
    ax[0][0].set_ylabel("Mode Strength")
    ax[0][0].set_title("Mode Strengths")

    # Plot normalized mode strengths
    ax[0][1].plot((eig_vals/S)*100, marker='o', color='purple')
    ax[0][1].set_xlabel("Mode Number")
    ax[0][1].set_ylabel("Normalized Mode Strength [%S]")
    ax[0][1].set_title("Normalized Mode Strengths")

    # Plot cumulative energy distribution
    ax[1][0].plot(np.cumsum(eig_vals), marker='o')
    ax[1][0].set_xlabel("Mode Number")
    ax[1][0].set_ylabel("Cumulative Energy")
    ax[1][0].set_title("Cumulative Energy Distribution")
    # Plot cumulative normalized energy distribution
    ax[1][1].plot(np.cumsum(eig_vals/S)*100, marker='o')
    ax[1][1].set_xlabel("Mode Number")
    ax[1][1].set_ylabel("Cumulative Normalized Energy [%S]")
    ax[1][1].set_title("Cumulative Normalized Energy Distribution")

    plt.tight_layout()
    plt.show()

def matprint(mat, fmt="g"):
    """
    Nicely prints a 2D matrix with aligned columns.

    Args:
        mat (np.ndarray): Matrix to print.
        fmt (str): Format string for each element.
    """
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for row in mat:
        for i, val in enumerate(row):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(val), end="  ")
        print("")
