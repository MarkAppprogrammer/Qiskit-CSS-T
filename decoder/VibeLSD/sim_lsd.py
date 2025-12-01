import galois
import numpy as np
import scipy.sparse
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from ldpc.bplsd_decoder import BpLsdDecoder
from mpi4py import MPI
import pandas as pd

sys.path.append('../../doubling-CSST/')
from convert_alist import readAlist
alistDirPath = "../../doubling-CSST/alistMats/GO03_self_dual/"

length_dist_dict = {4:2, 6:2, 8:2, 10:2, 12:4, 14:4, 16:4, 18:4, 20:4, 22:6, 24:6, 26:6, 28:6, 30:6, 32:8, 34:6, 36:8, 38:8, 40:8, 42:8, 44:8, 46:8, 48:8, 50:8, 52:10, 54:8, 56:10, 58:10, 60:12, 62:10, 64:10}

# --- Setup & Matrix Construction Functions ---
# self_dual_H(n): Constructs the self-dual parity check matrix H from .alist files, returning a sparse CSR matrix.
# initialize_decoder(...): Configures the decoder instance (Hyperion/MWPF or BP+LSD) based on the parity check matrix and error parameters.

# --- Noise Simulation Functions ---
# generate_errors(...): Simulates the physical noise channel by generating random X, Y, and Z error patterns based on probability and bias.

# --- Core Simulation Logic ---
# simulate_single_shot(...): Executes a full QEC cycle: calculates syndromes, runs the decoder, applies corrections, and detects logical failures.

# --- Orchestration (MPI) ---
# run_simulation(...): Drivers the MPI simulation. Distributes shots across ranks, iterates over code sizes/error rates, and reduces (aggregates) error counts to Rank 0.

# --- Data I/O & Visualization ---
# save_simulation_data(...): Serializes the simulation results (ns, ps, error rates) into a compressed NumPy (.npz) archive for later analysis.
# plot_combined_results(...): Visualizes the results by generating a 3-panel plot (Total, X, Z logical errors) and saving it as a PNG image.

def self_dual_H(n):
    d = length_dist_dict[n]
    F2 = galois.GF(2)
    alistFilePath = alistDirPath + "n" + str(n) + "_d" + str(d) + ".alist"
    GenMat = F2(readAlist(alistFilePath))
    G_punctured = GenMat[:, :-1]
    H_punctured = G_punctured.null_space()
    H = scipy.sparse.csr_matrix(np.array(H_punctured, dtype=int))
    return H

def initialize_decoder(Hx, Hz, error_rate, lsd_order, max_iter):
    bp_lsd_x = BpLsdDecoder(
        Hx,
        error_rate=error_rate,
        bp_method='product_sum',
        max_iter=max_iter,
        schedule='parallel',
        lsd_method='lsd_cs',
        lsd_order=lsd_order
    )
    bp_lsd_z = BpLsdDecoder(
        Hz,
        error_rate=error_rate,
        bp_method='product_sum',
        max_iter=max_iter,
        schedule='parallel',
        lsd_method='lsd_cs',
        lsd_order=lsd_order
    )
    return bp_lsd_x, bp_lsd_z

def generate_errors(num_shots, H, error_probability, bias_factor):
    rZ = bias_factor / (1 + bias_factor)
    rX = rY = (1 - rZ) / 2
    error_x = np.random.choice([0, 1], size=(num_shots, H.shape[1]), p=[1 - rX * error_probability, rX * error_probability])
    error_y = np.random.choice([0, 1], size=(num_shots, H.shape[1]), p=[1 - rY * error_probability, rY * error_probability])
    error_z = np.random.choice([0, 1], size=(num_shots, H.shape[1]), p=[1 - rZ * error_probability, rZ * error_probability])
    return error_x, error_y, error_z

def simulate_single_shot(Hx, Hz, Lx, Lz, error_x, error_y, error_z, decoder_x, decoder_z):
    error_x = (error_x + error_y) % 2
    error_z = (error_z + error_y) % 2
    
    syndromes_x = (error_z @ Hx.T) % 2
    syndromes_z = (error_x @ Hz.T) % 2
    
    correction_z = decoder_x.decode(syndromes_x)
    correction_x = decoder_z.decode(syndromes_z)
    
    residual_error_z = (correction_z + error_z) % 2
    residual_error_x = (correction_x + error_x) % 2
    
    logical_fail_x = (residual_error_z @ Lx.T) % 2
    logical_fail_z = (residual_error_x @ Lz.T) % 2
    
    return int(logical_fail_x.item()), int(logical_fail_z.item())

def run_simulation(ns, ps, bias_factor, max_iter, lsd_order, total_shots, comm, rank, size):
    # Split shots among ranks
    shots_per_proc = total_shots // size
    if rank == 0:
        shots_per_proc += (total_shots % size)

    results = {"total": [], "x": [], "z": []}

    for n in ns:
        if rank == 0:
            print(f"Simulating for n={n}...", flush=True)

        Hx = Hz = self_dual_H(n)
        Lx = Lz = np.ones((1, Hx.shape[1]), dtype=int)
        
        n_log_errors_total = []
        n_log_errors_x = []
        n_log_errors_z = []

        for error_rate in ps:
            bp_lsd_x, bp_lsd_z = initialize_decoder(Hx, Hz, error_rate, lsd_order, max_iter)
            error_x_local, error_y_local, error_z_local = generate_errors(shots_per_proc, Hx, error_rate, bias_factor)

            local_count_x = 0
            local_count_z = 0
            local_count_total = 0

            for i in range(shots_per_proc):
                fail_x, fail_z = simulate_single_shot(
                    Hx, Hz, Lx, Lz, 
                    error_x_local[i], error_y_local[i], error_z_local[i], 
                    bp_lsd_x, bp_lsd_z
                )
                if fail_x: local_count_x += 1
                if fail_z: local_count_z += 1
                if fail_x or fail_z: local_count_total += 1
            
            # Reduce to Rank 0
            total_x = comm.reduce(local_count_x, op=MPI.SUM, root=0)
            total_z = comm.reduce(local_count_z, op=MPI.SUM, root=0)
            total_err = comm.reduce(local_count_total, op=MPI.SUM, root=0)

            if rank == 0:
                n_log_errors_total.append(total_err / total_shots)
                n_log_errors_x.append(total_x / total_shots)
                n_log_errors_z.append(total_z / total_shots)
        
        if rank == 0:
            results["total"].append(n_log_errors_total)
            results["x"].append(n_log_errors_x)
            results["z"].append(n_log_errors_z)

    return results

def save_simulation_data(ns, ps, results, timestamp):
    filename = f"bplsd_mpi_data_{timestamp}.csv"
    data_to_save = []
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            row = {
                'n': n,
                'p': p,
                'total_error_rate': results['total'][i][j],
                'x_error_rate': results['x'][i][j],
                'z_error_rate': results['z'][i][j]
            }
            data_to_save.append(row)
    
    df = pd.DataFrame(data_to_save)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def plot_combined_results(ns, ps, bias_factor, results, timestamp):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    titles = ["Total Logical Error", "Logical X-side Error", "Logical Z-side Error"]
    keys = ["total", "x", "z"]
    
    for i, ax in enumerate(axes):
        key = keys[i]
        data_matrix = results[key]
        for j, n in enumerate(ns):
            ax.plot(ps, data_matrix[j], marker='o', linestyle='-', label=f'n={n}')
        ax.plot(ps, ps, linestyle='--', color='black', label='Break-even (y=x)', alpha=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Physical Error Rate (p)')
        ax.set_title(titles[i])
        ax.grid(True, which="both", ls="--", alpha=0.4)
        if i == 0: ax.set_ylabel('Logical Error Rate')
        if i == 2: ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Code Size")

    plt.suptitle(f'BP+LSD Logical Error Rates (Bias Factor: {bias_factor})', fontsize=16)
    plot_filename = f"bplsd_mpi_plot_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Plot saved as {plot_filename}")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Simulation parameters
    ns = [4, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64]
    ps = np.logspace(-3, -1, 20).tolist()
    lsd_order = 20
    max_iter = 1000
    bias_factor = 0.0
    num_shots = 1000_000

    seed = int(datetime.now().timestamp()) + rank * num_shots
    np.random.seed(seed)

    results = run_simulation(ns, ps, bias_factor, max_iter, lsd_order, num_shots, comm, rank, size)

    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_simulation_data(ns, ps, results, timestamp)
        plot_combined_results(ns, ps, bias_factor, results, timestamp)

if __name__ == "__main__":
    main()