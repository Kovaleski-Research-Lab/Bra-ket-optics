#
import numpy as np

def create_points(config: dict) -> np.ndarray:
    """
    Creates a grid of points based on the configuration parameters.

    Args:
        config (dict): Configuration dictionary containing grid parameters.
    Returns:
        np.ndarray: Array of points in the grid.
    """

    axis = config['axis']

    center_x, center_y, center_z = config.get('center', (0, 0, 0))
    print(f"Creating points in the {axis} plane with center at ({center_x}, {center_y}, {center_z})")
    
    try:
        if axis == 'xy':
            Lx = config['Lx']
            Ly = config['Ly']
            Nx = config['Nx']
            Ny = config['Ny']
            dx = Lx / (Nx - 1) if Nx > 1 else Lx
            dy = Ly / (Ny - 1) if Ny > 1 else Ly
            s = np.asarray([(i * dx, j * dy, center_z) for i in range(Nx) for j in range(Ny)])
            s[:, 0] -= Lx/2
            s[:, 1] -= Ly/2
        elif axis == 'yz':
            Ly = config['Ly']
            Lz = config['Lz']
            Ny = config['Ny']
            Nz = config['Nz']
            dy = Ly / (Ny - 1) if Ny > 1 else Ly
            dz = Lz / (Nz - 1) if Nz > 1 else Lz
            s = np.asarray([(center_x, j * dy, i * dz) for i in range(Nz) for j in range(Ny)])
            s[:, 1] -= Ly/2
            s[:, 2] -= Lz/2
        elif axis == 'xz':
            Lx = config['Lx']
            Lz = config['Lz']
            Nx = config['Nx']
            Nz = config['Nz']
            dx = Lx / (Nx - 1) if Nx > 1 else Lx
            dz = Lz / (Nz - 1) if Nz > 1 else Lz
            s = np.asarray([(i * dx, center_y, j * dz) for i in range(Nx) for j in range(Nz)])
            s[:, 0] -= Lx/2
            s[:, 2] -= Lz/2
        else:
            raise ValueError("Axis must be 'xy', 'yz', or 'xz'.")
    except KeyError as e:
        raise KeyError(f"Missing configuration parameter: {e}")
    except TypeError as e:
        raise TypeError(f"Invalid type for configuration parameter: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while creating points: {e}")

    s = np.asarray(s, dtype=float)
    return s


def create_evaluation_points(config: dict):

    axis = config['axis']
    discretization = config.get('discretization', 100)
    center_x, center_y, center_z = config.get('center', (0, 0, 0))
    print(f"Creating evaluation points in the {axis} plane with center at ({center_x}, {center_y}, {center_z})")

    if axis == 'xy':
        Lx = config['Lx']
        Ly = config['Ly']
        z = config.get('z', center_z)
        x = np.linspace(center_x - Lx/2, center_x + Lx/2, discretization)
        y = np.linspace(center_y - Ly/2, center_y + Ly/2, discretization)
        xx, yy,zz = np.meshgrid(x, y, z, indexing='ij')
        r = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    elif axis == 'yz':
        Ly = config['Ly']
        Lz = config['Lz']
        x = config.get('x', center_x)
        y = np.linspace(center_y - Ly/2, center_y + Ly/2, discretization)
        z = np.linspace(center_z - Lz/2, center_z + Lz/2, discretization)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        r = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    elif axis == 'xz':
        Lx = config['Lx']
        Lz = config['Lz']
        y = config.get('y', center_y)
        x = np.linspace(center_x - Lx/2, center_x + Lx/2, discretization)
        z = np.linspace(center_z - Lz/2, center_z + Lz/2, discretization)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        r = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    else:
        raise ValueError("Axis must be 'xy', 'yz', or 'xz'.")

    return xx, yy, zz, r


