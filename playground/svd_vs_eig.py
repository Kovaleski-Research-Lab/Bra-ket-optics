import os
import sys
import numpy as np
import scipy 


sys.path.append('../')

from src.eig import *
from src.geometry import *
from src.utils import *


if __name__ == "__main__":

    wavelength = 1
    source_params = {
        'geometry': 'line',
        'axis': 'x',
        'Nx': 3,
        'Ny': None,
        'Nz': None,
        'Lx': 2,
        'Ly': None,
        'Lz': None, 
        'center': [0, 0, 0],
    }
    
    receiver_params = {
        'geometry': 'line',
        'axis': 'x',
        'Nx': 3,
        'Ny': None,
        'Nz': None,
        'Lx': 2,
        'Ly': None,
        'Lz': None, 
        'center': [0, 0, 5],
    }


    source_points = create_points(source_params)
    receiver_points = create_points(receiver_params)
    
    k = 2 * np.pi / wavelength

    Gsr = free_space_transfer_function(source_points, receiver_points, k, method='direct')

    S = sum_rule(Gsr)

    # First, SVD on the Gsr matrix
    U, s, Vh = np.linalg.svd(Gsr, full_matrices=True)

    # Now, eigenvector/eigenvalue decomposition on Gsrd_Gsr and Gsr_Gsrd
    Gsrd_Gsr = Gsr.conj().T @ Gsr
    eig_vals_source, eig_vect_source = np.linalg.eigh(Gsrd_Gsr)
    eig_vals_source = np.flip(eig_vals_source)
    eig_vect_source = np.flip(eig_vect_source, axis=-1)
    
    Gsr_Gsrd = Gsr @ Gsr.conj().T
    eig_vals_receiver, eig_vect_receiver = np.linalg.eigh(Gsr_Gsrd)
    eig_vals_receiver = np.flip(eig_vals_receiver)
    eig_vect_receiver = np.flip(eig_vect_receiver, axis=-1)

    Vh = Vh.conj().T  # Make Vh a column matrix

    # Check alignment (up to sign/phase)
    for i in range(len(s)):  
        angle_v = np.vdot(Vh[:, i], eig_vect_source[:, i])
        angle_u = np.vdot(U[:, i], eig_vect_receiver[:, i])
        print(f"Right vector alignment {i}: {np.abs(angle_v)}")
        print(f"Left vector alignment {i}: {np.abs(angle_u)}")
    from IPython import embed; embed()


