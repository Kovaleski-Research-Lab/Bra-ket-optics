import numpy as np
import sys
sys.path.append('../')
from src.eig import (
    forward_projection_direct,
    inverse_projection_direct,
    free_space_transfer_function,
    calculate_modes,
    forward_projection_eig,
    inverse_projection_eig
)
from src.geometry import create_points
from src.utils import scale_config
import itertools
import pandas as pd
from tqdm import tqdm

def generate_config(Lx, Ly, Lz, Nx, Ny, Nz, center0, center1, wavelength):
    """
    Generates a configuration dictionary for testing. 
    """
    config = {
        'source': {
            'geometry': 'plane',
            'axis': 'xy',
            'Lx': Lx,
            'Ly': Ly,
            'Lz': Lz,
            'Nx': Nx,
            'Ny': Ny,
            'Nz': Nz,
            'center': center0
        },
        'receiver': {
            'geometry': 'plane',
            'axis': 'xy',
            'Lx': Lx,
            'Ly': Ly,
            'Lz': Lz,
            'Nx': Nx,
            'Ny': Ny,
            'Nz': Nz,
            'center': center1
        },
        'wavelength': wavelength
    }

    return config


# The goal for this file is to compare functions as we span physical dimensions
# from super-wavelength to sub-wavelength. This includes the extent of the planes
# (Lx, Ly) as well as the propagation distance (center[2] of the 2nd plane assuming the center of the first plane is [0,0,0]).
# We might want to also try when dx and dy are either super-wavelength or sub-wavelength.
# I think the only way to test this is with a full round trip ( forward + inverse ).
# Then compare the result to the input field.


# Helper function to create a gaussian distribution
def create_gaussian(nx: int, ny: int, sigma: float = 0.3) -> np.ndarray:
        """Create Gaussian field pattern."""
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        return np.exp(-(X**2 + Y**2) / (2 * sigma**2)) * (1 + 1j)


if __name__ == "__main__":

    wavelength = 2.0

    Lx_range = np.round(np.geomspace(0.1 * wavelength, 50 * wavelength, 12), 4)
    Ly_range = np.round(np.geomspace(0.1 * wavelength, 50 * wavelength, 12), 4)
    Lz = 0

    Nx = 25
    Ny = 25

    center0 = np.array([0.0, 0.0, 0.0])
    center1 = np.array([0.0, 0.0, 0.0])
    center1_z_range = np.round(np.geomspace(0.1 * wavelength, 50 * wavelength, 12), 4)

    
    # Dictionary to hold errors for each method combination
    errors = {'eig_direct': {}, 'eig_input': {}, 'direct_input': {}}

    results = []
    # Iterate over all combinations of Lx, Ly, and center1_z
    experiments = list(itertools.product(Lx_range, Ly_range, center1_z_range))
    for Lx, Ly, center1_z in tqdm(experiments):
        c1 = center1.copy()
        c1[2] = center1_z
        config = generate_config(Lx, Ly, None, Nx, Ny, None, center0.copy(), c1, wavelength)
        config = scale_config(config)

        # Create source and receiver points
        source_points = create_points(config['source'])
        receiver_points = create_points(config['receiver'])

        # Create a sample input field (Gaussian)
        input_field = create_gaussian(Nx, Ny).reshape(-1, 1)
        input_field = input_field / np.linalg.norm(input_field)

        # Wave number
        new_wavelength = config['wavelength']
        k = 2 * np.pi / new_wavelength

        # Calculate Gsr
        Gsr = free_space_transfer_function(source_points, receiver_points, k, method='blockwise')

        # Calculate modes
        eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr, normalize=False)

        # Run the direct method
        direct_forward = forward_projection_direct(input_field, Gsr)
        direct_inverse = inverse_projection_direct(direct_forward, Gsr)

        # Run the eig method
        eig_forward = forward_projection_eig(input_field, eig_vals, eig_vect_source, eig_vect_receiver)
        eig_inverse = inverse_projection_eig(eig_forward, eig_vals, eig_vect_source, eig_vect_receiver)

        # Compare results
        eig_direct_forward_error = np.linalg.norm(direct_forward - eig_forward) / np.linalg.norm(direct_forward)
        eig_direct_inverse_error = np.linalg.norm(direct_inverse - eig_inverse) / np.linalg.norm(direct_inverse)

        eig_direct_error = np.linalg.norm(eig_forward - direct_forward) / np.linalg.norm(direct_forward)
        eig_input_error = np.linalg.norm(eig_forward - input_field) / np.linalg.norm(input_field)
        direct_input_error = np.linalg.norm(direct_forward - input_field) / np.linalg.norm(input_field)
        results.append({
            'Lx': Lx, 'Ly': Ly, 'z': center1_z,
            'eig_direct_forward_error': eig_direct_forward_error,
            'eig_direct_inverse_error': eig_direct_inverse_error,
            'eig_direct_error': eig_direct_error,
            'eig_input_error': eig_input_error,
            'direct_input_error': direct_input_error
        })

    df = pd.DataFrame(results)
    df.to_csv('experiment_metrics.csv', index=False)


