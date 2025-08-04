import numpy as np
import pytest
import sys
from src.geometry import create_points

def test_point():
    config = {
        'geometry': 'point',
        'center': (1.0, 2.0, 3.0)
    }
    pts = create_points(config)
    expected = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(pts, expected)

def test_line_x_axis():
    config = {
        'geometry': 'line',
        'axis': 'x',
        'Lx': 4.0,
        'Nx': 5,
        'center': (0.0, 0.0, 0.0)
    }
    pts = create_points(config)
    expected_x = np.linspace(-2.0, 2.0, 5)
    np.testing.assert_allclose(pts[:, 0], expected_x)
    assert np.all(pts[:, 1] == 0.0)
    assert np.all(pts[:, 2] == 0.0)

def test_plane_xy():
    config = {
        'geometry': 'plane',
        'axis': 'xy',
        'Lx': 2.0,
        'Ly': 2.0,
        'Nx': 3,
        'Ny': 3,
        'center': (0.0, 0.0, 1.0)
    }
    pts = create_points(config)
    assert pts.shape == (9, 3)
    assert np.allclose(pts[:, 2], 1.0)
    xs = np.linspace(-1.0, 1.0, 3)
    ys = np.linspace(-1.0, 1.0, 3)
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            np.testing.assert_allclose(pts[idx], [xs[i], ys[j], 1.0])

def test_volume_centering():
    config = {
        'geometry': 'volume',
        'Lx': 2.0,
        'Ly': 2.0,
        'Lz': 2.0,
        'Nx': 2,
        'Ny': 2,
        'Nz': 2,
        'center': (1.0, 2.0, 3.0)
    }
    pts = create_points(config)
    assert pts.shape == (8, 3)
    expected_x = np.linspace(-1.0, 1.0, 2) + 1.0
    expected_y = np.linspace(-1.0, 1.0, 2) + 2.0
    expected_z = np.linspace(-1.0, 1.0, 2) + 3.0
    for x in expected_x:
        for y in expected_y:
            for z in expected_z:
                assert any(np.allclose(p, [x, y, z]) for p in pts)

def test_invalid_geometry():
    config = {'geometry': 'cube'}
    with pytest.raises(ValueError):
        create_points(config)

def test_missing_axis_for_plane():
    config = {
        'geometry': 'plane',
        # 'axis' is missing
        'Lx': 1.0, 'Ly': 1.0, 'Nx': 2, 'Ny': 2,
        'center': (0, 0, 0)
    }
    with pytest.raises(KeyError):
        create_points(config)

