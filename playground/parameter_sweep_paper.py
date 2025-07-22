#
import yaml
from src.utils import *
from src.eig import free_space_transfer_function, calculate_modes
from src.plotting import plot_modes, plot_mode_strengths
from src.geometry import create_points, create_evaluation_points
import pickle

if __name__ == "__main__":
    config = yaml.safe_load(open("parameter_sweep_config.yaml", "r"))

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
    max_components = config.get('max_components', None)
    eig_vals, eig_vect_normalized = calculate_modes(Gsr, normalize=False, max_components=max_components)

    # Save the eigenvectors/values
    pickle.dump(eig_vals, open('eigen_values_sweep.pkl', 'wb'))
    pickle.dump(eig_vect_normalized, open('eigen_vectors_sweep.pkl', 'wb'))

    # Plot mode strengths
    plot_mode_strengths(eig_vals, S)

    # Create evaluation points
    xx,yy,zz,evaluation_points = create_evaluation_points(config['plot_plane'])
    print("Evaluation points shape:", evaluation_points.shape)
    from IPython import embed; embed()

    # Evaluate modes at the evaluation points
    modes = evaluate_modes(eig_vect_normalized, source_points, evaluation_points, wavenumber, z_normalize=True, method='parallel')

    # Save the modes
    pickle.dump(modes, open('evaluated_modes_sweep.pkl', 'wb'))

    # Plot modes
    plot_modes(xx, yy, zz, modes, plane=config['plot_plane']['axis'], y=config['plot_plane'].get('y_idx'))

