import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('../')  # Adjust this path as necessary
from src.eig import (
    free_space_transfer_function,
    calculate_modes,
    forward_projection_direct,
    inverse_projection_direct,
    forward_projection_eig,
    inverse_projection_eig
)

def benchmark_function(func, *args, repeat=3, **kwargs):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)

def generate_points(N, dtype=np.float64):
    # Create 3D grid points
    points = np.random.uniform(-1, 1, size=(N, 3)).astype(dtype)
    return points

def run_timing_experiments(sizes, dtypes, method='direct'):
    results = []

    for dtype in dtypes:
        print(f"\nRunning tests with dtype: {dtype.__name__}")
        for N in sizes:
            print(f"  Matrix size: {N}x{N}")
            s = generate_points(N).astype(np.float64)
            r = generate_points(N).astype(np.float64)
            k = 2 * np.pi  # Wavenumber

            # Allocate source field with correct dtype
            source_field = np.random.randn(N, 1).astype(dtype) + 1j * np.random.randn(N, 1).astype(dtype)

            # Free space transfer function
            t_tf, _ = benchmark_function(
                free_space_transfer_function, s, r, k, normalize=False, method=method
            )

            # Compute Gsr for further steps
            Gsr = free_space_transfer_function(s, r, k, normalize=False, method=method).astype(dtype)

            # Calculate modes
            t_modes, _ = benchmark_function(calculate_modes, Gsr)

            eig_vect_receiver, eig_vals, eig_vect_source = calculate_modes(Gsr)

            # Direct forward projection
            t_fwd_direct, _ = benchmark_function(forward_projection_direct, source_field, Gsr)

            # Direct inverse projection
            receiver_field = forward_projection_direct(source_field, Gsr)
            t_inv_direct, _ = benchmark_function(inverse_projection_direct, receiver_field, Gsr)

            # Mode-based forward projection
            t_fwd_eig, _ = benchmark_function(forward_projection_eig, source_field, eig_vals, eig_vect_source, eig_vect_receiver)

            # Mode-based inverse projection
            t_inv_eig, _ = benchmark_function(inverse_projection_eig, receiver_field, eig_vals, eig_vect_source, eig_vect_receiver)

            results.append({
                'N': N,
                'dtype': dtype.__name__,
                'method': method,
                't_transfer_function': t_tf,
                't_modes': t_modes,
                't_forward_direct': t_fwd_direct,
                't_inverse_direct': t_inv_direct,
                't_forward_eig': t_fwd_eig,
                't_inverse_eig': t_inv_eig
            })
    return results

def plot_timing(results, key, title):
    df = pd.DataFrame(results)
    for dtype in df['dtype'].unique():
        subset = df[df['dtype'] == dtype]
        plt.plot(subset['N'], subset[key], label=f"{dtype}")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    sizes = [i * 10 for i in range(10)]  # Increase depending on machine power
    dtypes = [np.complex64, np.complex128]
    results_direct = run_timing_experiments(sizes, dtypes, method='direct')
    results_block = run_timing_experiments(sizes, dtypes, method='blockwise')

    plot_timing(results_direct, 't_transfer_function', "Transfer Function Time (Direct)")
    plot_timing(results_block, 't_transfer_function', "Transfer Function Time (Blockwise)")
    plot_timing(results_direct, 't_modes', "Mode Calculation Time")
    plot_timing(results_direct, 't_forward_direct', "Forward Projection (Direct)")
    plot_timing(results_direct, 't_forward_eig', "Forward Projection (Eigenspace)")



