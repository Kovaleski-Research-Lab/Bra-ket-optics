import yaml
from src.utils import *
from src.eig import free_space_transfer_function, calculate_modes
from src.plotting import plot_modes, plot_mode_strengths
from src.geometry import create_points, create_evaluation_points


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml", "r"))

    source_points = create_points(config['source'])
    receiver_points = create_points(config['receiver'])


    print("Source points shape:", source_points.shape)
    print("Receiver points shape:", receiver_points.shape)

    wavelength = config['wavelength']
    wavenumber = 2 * np.pi / wavelength

    
    # Compute transfer matrix
    Gsr = free_space_transfer_function(source_points, receiver_points, wavenumber, method='blockwise')

    # Compute normalization constant
    S = sum_rule(Gsr)
    print("S:", S)

    # Calculate modes
    eig_vals, eig_vect_normalized = calculate_modes(Gsr, normalize=False)

    # Plot mode strengths
    plot_mode_strengths(eig_vals, S)


    # Create evaluation points
    xx,yy,zz,evaluation_points = create_evaluation_points(config['plot_plane'])
    print("Evaluation points shape:", evaluation_points.shape)

    # Evaluate modes at the evaluation points
    modes = evaluate_modes(eig_vect_normalized, source_points, evaluation_points, wavenumber, z_normalize=True, method='vectorized')

    # Plot modes
    plot_modes(xx, yy, zz, modes, plane=config['plot_plane']['axis'], z=config['plot_plane'].get('z_idx'))

