import numpy as np
import itertools
import random  # use random, not np.random here
import copy

def generate_configs(num_samples=20, seed=42):
    geometries = ['plane']
    axes = ['xy', 'xz', 'yz']
    lengths = [100, 200, 300]
    num_points = [10, 20, 50]
    centers = [(0.0, 0.0, 0.0), (100.0, 100.0, 100.0), (200.0, 200.0, 200.0)]
    offset_distance = 750.0
    all_configs = []

    def build_entry(geometry, axis, L_vals, N_vals, center):
        entry = {
            'geometry': geometry,
            'axis': axis,
            'Lx': None,
            'Ly': None,
            'Lz': None,
            'Nx': None,
            'Ny': None,
            'Nz': None,
            'center': list(center)
        }
        if 'x' in axis:
            entry['Lx'] = L_vals[0]
            entry['Nx'] = N_vals[0]
        if 'y' in axis:
            entry['Ly'] = L_vals[1]
            entry['Ny'] = N_vals[1]
        if 'z' in axis:
            entry['Lz'] = L_vals[2]
            entry['Nz'] = N_vals[2]
        return entry

    def get_optical_axis(axis):
        axis_map = {
            'x': 'x',
            'y': 'y',
            'z': 'z',
            'xy': 'z',
            'xz': 'y',
            'yz': 'x'
        }
        return axis_map[axis]

    def offset_center(center, axis):
        optical_axis = get_optical_axis(axis)
        axis_to_index = {'x': 0, 'y': 1, 'z': 2}
        idx = axis_to_index[optical_axis]
        center = list(center)
        center[idx] += offset_distance
        return tuple(center)

    # Generate all valid configs
    for geometry in geometries:
        for axis in axes:
            for Lx, Ly, Lz in itertools.product(lengths, repeat=3):
                for Nx, Ny, Nz in itertools.product(num_points, repeat=3):
                    for center_src in centers:
                        center_rcv = offset_center(center_src, axis)
                        if center_rcv == center_src:
                            continue
                        source = build_entry(geometry, axis, (Lx, Ly, Lz), (Nx, Ny, Nz), center_src)
                        receiver = build_entry(geometry, axis, (Lx, Ly, Lz), (Nx, Ny, Nz), center_rcv)
                        all_configs.append({'source': source, 'receiver': receiver})

    # Sample randomly
    random.seed(seed)
    if len(all_configs) <= num_samples:
        return all_configs
    return random.sample(all_configs, num_samples)

