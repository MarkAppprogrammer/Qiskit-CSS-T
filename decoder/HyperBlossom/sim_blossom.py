import galois
import numpy as np
import scipy.sparse
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from mwpf import HyperEdge, SolverInitializer, SolverSerialJointSingleHair, SyndromePattern
from mpi4py import MPI
import pandas as pd

sys.path.append('../../doubling-CSST/')
from convert_alist import readAlist
alistDirPath = "../../doubling-CSST/alistMats/GO03_self_dual/"

DEFAULT_WEIGHT = 100

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

def initialize_decoder(matrix, weight):
    mat_np = matrix.toarray().astype(int)
    num_rows, num_cols = mat_np.shape
    edges_list = []
    
    for c in range(num_cols):
        check_indices = np.nonzero(mat_np[:, c])[0].tolist()
        if len(check_indices) > 0:
            edges_list.append(HyperEdge(check_indices, weight))
        else:
            edges_list.append(HyperEdge([], weight))

    initializer = SolverInitializer(num_rows, edges_list)
    hyperion = SolverSerialJointSingleHair(initializer)
    return hyperion

def generate_errors(num_shots, H, error_probability, bias_factor):
    rZ = bias_factor / (1 + bias_factor)
    rX = rY = (1 - rZ) / 2

    error_x = np.random.choice(
        [0, 1], size=(num_shots, H.shape[1]), p=[1 - rX * error_probability, rX * error_probability]
    )
    error_y = np.random.choice(
        [0, 1], size=(num_shots, H.shape[1]), p=[1 - rY * error_probability, rY * error_probability]
    )
    error_z = np.random.choice(
        [0, 1], size=(num_shots, H.shape[1]), p=[1 - rZ * error_probability, rZ * error_probability]
    )
    return error_x, error_y, error_z

def simulate_single_shot(Hx, Hz, Lx, Lz, error_x, error_y, error_z, decoder):
    error_x = (error_x + error_y) % 2
    error_z = (error_z + error_y) % 2
    
    syndromes_x = (error_z @ Hx.T) % 2
    syndromes_z = (error_x @ Hz.T) % 2
    syndromes = np.hstack([syndromes_x, syndromes_z]) 
    syndrome_indices = np.nonzero(syndromes)[0].tolist()
    n_qubits = Hx.shape[1]
    
    if not syndrome_indices:
        correction_vector = np.zeros(2 * n_qubits, dtype=int)
    else:
        decoder.solve(SyndromePattern(syndrome_indices))
        subgraph_indices = decoder.subgraph()
        correction_vector = np.zeros(2 * n_qubits, dtype=int)
        correction_vector[subgraph_indices] = 1

    correction_z = correction_vector[:n_qubits]
    correction_x = correction_vector[n_qubits:]

    residual_error_z = (correction_z + error_z) % 2
    residual_error_x = (correction_x + error_x) % 2

    logical_fail_x = (residual_error_x @ Lx.T) % 2
    logical_fail_z = (residual_error_z @ Lz.T) % 2

    return int(logical_fail_x.item()), int(logical_fail_z.item())

def run_simulation(ns, ps, bias_factor, total_shots, comm, rank, size):
    shots_per_proc = total_shots // size
    if rank == 0:
        shots_per_proc += (total_shots % size)

    results = {"total": [], "x": [], "z": []}

    for n in ns:
        if rank == 0:
            print(f"Simulating for n={n}...", flush=True)
            
        Hx = Hz = self_dual_H(n)
        Lx = Lz = np.ones((1, Hx.shape[1]), dtype=int)
        H_CSS = scipy.sparse.block_diag((Hx, Hz))
        hyperion = initialize_decoder(H_CSS, DEFAULT_WEIGHT)
        
        n_log_errors_total = []
        n_log_errors_x = []
        n_log_errors_z = []

        for error_rate in ps:
            # Generate local errors for this process
            error_x_local, error_y_local, error_z_local = generate_errors(shots_per_proc, Hx, error_rate, bias_factor)

            local_count_x = 0
            local_count_z = 0
            local_count_total = 0

            for i in range(shots_per_proc):
                fail_x, fail_z = simulate_single_shot(
                    Hx, Hz, Lx, Lz, 
                    error_x_local[i], error_y_local[i], error_z_local[i], 
                    hyperion
                )
                if fail_x: local_count_x += 1
                if fail_z: local_count_z += 1
                if fail_x or fail_z: local_count_total += 1
            
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
    filename = f"hyperion_mpi_data_{timestamp}.csv"
    
    data_rows = []
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            row = {
                'n': n,
                'p': p,
                'total_logical_error_rate': results['total'][i][j],
                'x_logical_error_rate': results['x'][i][j],
                'z_logical_error_rate': results['z'][i][j]
            }
            data_rows.append(row)
            
    df = pd.DataFrame(data_rows)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def plot_combined_results(ns, ps, bias_factor, results, timestamp):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    titles = ["Total Logical Error", "Logical X Error", "Logical Z Error"]
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

    plt.suptitle(f'Hyperion Logical Error Rates (Bias Factor: {bias_factor})', fontsize=16)
    plot_filename = f"hyperion_mpi_plot_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Plot saved as {plot_filename}")

def main():
    # MPI Initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Simulation parameters
    ns = [4, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64]
    ps = np.logspace(-3, -1, 20).tolist()
    bias_factor = 0.0
    num_shots = 1000_000

    seed = int(datetime.now().timestamp()) + rank * num_shots
    np.random.seed(seed)

    results = run_simulation(ns, ps, bias_factor, num_shots, comm, rank, size)

    # Only Rank 0 saves and plots
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_simulation_data(ns, ps, results, timestamp)
        plot_combined_results(ns, ps, bias_factor, results, timestamp)

if __name__ == "__main__":
    main()