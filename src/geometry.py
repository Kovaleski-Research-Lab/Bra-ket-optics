import numpy as np

def create_point(config: dict) -> np.ndarray:
    """Returns a single 3D point."""
    center = np.array(config.get('center', (0, 0, 0)), dtype=float)
    return center.reshape(1, 3)


def create_line(config: dict) -> np.ndarray:
    axis = config['axis']
    center = np.array(config.get('center', (0, 0, 0)), dtype=float)
    
    if axis == 'x':
        L, N = config['Lx'], config['Nx']
        values = np.linspace(-L / 2, L / 2, N) + center[0]
        s = np.stack([values, np.full(N, center[1]), np.full(N, center[2])], axis=-1)

    elif axis == 'y':
        L, N = config['Ly'], config['Ny']
        values = np.linspace(-L / 2, L / 2, N) + center[1]
        s = np.stack([np.full(N, center[0]), values, np.full(N, center[2])], axis=-1)

    elif axis == 'z':
        L, N = config['Lz'], config['Nz']
        values = np.linspace(-L / 2, L / 2, N) + center[2]
        s = np.stack([np.full(N, center[0]), np.full(N, center[1]), values], axis=-1)

    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    
    return s.astype(float)


def create_plane(config: dict) -> np.ndarray:
    axis = config['axis']
    center = np.array(config.get('center', (0, 0, 0)), dtype=float)

    if axis == 'xy':
        Lx, Nx = config['Lx'], config['Nx']
        Ly, Ny = config['Ly'], config['Ny']
        x = np.linspace(-Lx / 2, Lx / 2, Nx) + center[0]
        y = np.linspace(-Ly / 2, Ly / 2, Ny) + center[1]
        xx, yy = np.meshgrid(x, y, indexing='ij')
        zz = np.full_like(xx, center[2])
        s = np.stack([xx, yy, zz], axis=-1)

    elif axis == 'yz':
        Ly, Ny = config['Ly'], config['Ny']
        Lz, Nz = config['Lz'], config['Nz']
        y = np.linspace(-Ly / 2, Ly / 2, Ny) + center[1]
        z = np.linspace(-Lz / 2, Lz / 2, Nz) + center[2]
        yy, zz = np.meshgrid(y, z, indexing='ij')
        xx = np.full_like(yy, center[0])
        s = np.stack([xx, yy, zz], axis=-1)

    elif axis == 'xz':
        Lx, Nx = config['Lx'], config['Nx']
        Lz, Nz = config['Lz'], config['Nz']
        x = np.linspace(-Lx / 2, Lx / 2, Nx) + center[0]
        z = np.linspace(-Lz / 2, Lz / 2, Nz) + center[2]
        xx, zz = np.meshgrid(x, z, indexing='ij')
        yy = np.full_like(xx, center[1])
        s = np.stack([xx, yy, zz], axis=-1)

    else:
        raise ValueError("Axis must be 'xy', 'yz', or 'xz'.")
    
    return s.reshape(-1, 3).astype(float)


def create_volume(config: dict) -> np.ndarray:
    Lx, Nx = config['Lx'], config['Nx']
    Ly, Ny = config['Ly'], config['Ny']
    Lz, Nz = config['Lz'], config['Nz']
    center = np.array(config.get('center', (0, 0, 0)), dtype=float)

    x = np.linspace(-Lx / 2, Lx / 2, Nx) + center[0]
    y = np.linspace(-Ly / 2, Ly / 2, Ny) + center[1]
    z = np.linspace(-Lz / 2, Lz / 2, Nz) + center[2]

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    s = np.stack([xx, yy, zz], axis=-1)
    
    return s.reshape(-1, 3).astype(float)


def create_points(config: dict) -> np.ndarray:
    """
    Creates a grid of points based on the configuration parameters.

    Args:
        config (dict): Configuration dictionary containing grid parameters.
    Returns:
        np.ndarray: Array of points in the grid.
    """
    geometry_type = config['geometry']
    
    if geometry_type == 'point':
        return create_point(config)
    elif geometry_type == 'line':
        return create_line(config)
    elif geometry_type == 'plane':
        return create_plane(config)
    elif geometry_type == 'volume':
        return create_volume(config)
    else:
        raise ValueError("Geometry must be 'point', 'line', 'plane', or 'volume'.")


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


