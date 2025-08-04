
import numpy as np
import pytest

from src.eig import (
    forward_projection,
    free_space_transfer_function
)

from src.geometry import create_points



@pytest.fixture
def test_config_01():
    return {
        'source': {
            'geometry': 'plane',
            'axis': 'xy',
            'Nx': 4,
            'Ny': 4,
            'Nz': None,
            'Lx': 4,
            'Ly': 4,
            'Lz': None,
            'center': [0, 0, 0]
        },
        'receiver': {
            'geometry': 'plane',
            'axis': 'xy',
            'Nx': 4,
            'Ny': 4,
            'Nz': None,
            'Lx': 4,
            'Ly': 4,
            'Lz': None,
            'center': [0, 0, 10]
        }
    }
