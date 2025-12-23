import galois
import numpy as np
import scipy.sparse
import scipy.stats
import sys
from datetime import datetime
from mwpf import HyperEdge, SolverInitializer, SolverSerialJointSingleHair, SyndromePattern
from mpi4py import MPI
import pandas as pd
import os
import time

sys.path.append('../../doubling-CSST/')
from convert_alist import readAlist

# --- Global Configuration ---
DEFAULT_WEIGHT = 100
ALPHA = 0.05

alistDirPath = "../../doubling-CSST/alistMats/QR_dual_containing/" 
length_dist_dict = {7:3, 17:5, 23:7, 47:11, 79:15, 103:19, 167:23} 

def H(n, if_self_dual=False):
	d = length_dist_dict[n]
	F2 = galois.GF(2)
	alistFilePath = alistDirPath + "n" + str(n) + "_d" + str(d) + ".alist"
	GenMat = F2(readAlist(alistFilePath))
	
	if if_self_dual:
		G_punctured = GenMat[:, :-1]
		H_punctured = G_punctured.null_space()
	else:
		H_punctured = GenMat.null_space() 

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
    error_x = np.random.choice([0, 1], size=(num_shots, H.shape[1]), p=[1 - rX * error_probability, rX * error_probability])
    error_y = np.random.choice([0, 1], size=(num_shots, H.shape[1]), p=[1 - rY * error_probability, rY * error_probability])
    error_z = np.random.choice([0, 1], size=(num_shots, H.shape[1]), p=[1 - rZ * error_probability, rZ * error_probability])
    return error_x, error_y, error_z

def simulate_single_shot(Hx, Hz, Lx, Lz, error_x, error_y, error_z, decoder):
    error_x = (error_x + error_y) % 2
    error_z = (error_z + error_y) % 2
    syndromes_x = (error_z @ Hx.T) % 2
    syndromes_z = (error_x @ Hz.T) % 2
    syndromes = np.hstack([syndromes_x, syndromes_z]) 
    syndrome_indices = np.nonzero(syndromes)[0].tolist()
    n_qubits = Hx.shape[1]
    objective_value = 0.0
    
    if not syndrome_indices:
        correction_vector = np.zeros(2 * n_qubits, dtype=int)
    else:
        decoder.solve(SyndromePattern(syndrome_indices))
        subgraph_indices = decoder.subgraph()
        _, bound = decoder.subgraph_range()
        s = str(bound.upper - bound.lower)
        objective_value = float(s.split('/')[0]) / float(s.split('/')[1]) if '/' in s else float(s)
        correction_vector = np.zeros(2 * n_qubits, dtype=int)
        correction_vector[subgraph_indices] = 1

    correction_z = correction_vector[:n_qubits]
    correction_x = correction_vector[n_qubits:]
    residual_error_z = (correction_z + error_z) % 2
    residual_error_x = (correction_x + error_x) % 2
    logical_fail_x = (residual_error_x @ Lx.T) % 2
    logical_fail_z = (residual_error_z @ Lz.T) % 2

    return int(logical_fail_x.item()), int(logical_fail_z.item()), objective_value

def calc_ci(k_arr, n):
    k = np.array(k_arr)
    p_hat = k / n
    lower = scipy.stats.beta.ppf(ALPHA / 2, k, n - k + 1)
    lower[k == 0] = 0.0 
    upper = scipy.stats.beta.ppf(1 - ALPHA / 2, k + 1, n - k)
    upper[k == n] = 1.0 
    err_low = p_hat - lower
    err_high = upper - p_hat
    return np.vstack((err_low, err_high))

def run_simulation(ns, ps, bias_factor, total_shots, comm, rank, size, filename):
    shots_per_proc = total_shots // size
    if rank == 0:
        shots_per_proc += (total_shots % size)

    for i, n in enumerate(ns):
        if rank == 0: print(f"Simulating for n={n}...", flush=True)
            
        Hx = Hz = H(n)
        Lx = Lz = np.ones((1, Hx.shape[1]), dtype=int)
        H_CSS = scipy.sparse.block_diag((Hx, Hz))
        hyperion = initialize_decoder(H_CSS, DEFAULT_WEIGHT)
        
        local_total_counts = np.zeros(len(ps), dtype=np.int64)
        local_x_counts = np.zeros(len(ps), dtype=np.int64)
        local_z_counts = np.zeros(len(ps), dtype=np.int64)
        local_cpu_times = np.zeros(len(ps), dtype=np.float64)
        local_obj_values = np.zeros(len(ps), dtype=np.float64)

        for idx, error_rate in enumerate(ps):
            start_time = time.time()
            error_x_local, error_y_local, error_z_local = generate_errors(shots_per_proc, Hx, error_rate, bias_factor)
            for shot_idx in range(shots_per_proc):
                fail_x, fail_z, obj_val = simulate_single_shot(Hx, Hz, Lx, Lz, error_x_local[shot_idx], error_y_local[shot_idx], error_z_local[shot_idx], hyperion)
                if fail_x: local_x_counts[idx] += 1
                if fail_z: local_z_counts[idx] += 1
                if fail_x or fail_z: local_total_counts[idx] += 1
                local_obj_values[idx] += obj_val
            end_time = time.time()
            local_cpu_times[idx] = end_time - start_time
        
        global_total_counts = np.zeros(len(ps), dtype=np.int64) if rank == 0 else None
        global_x_counts = np.zeros(len(ps), dtype=np.int64) if rank == 0 else None
        global_z_counts = np.zeros(len(ps), dtype=np.int64) if rank == 0 else None
        global_cpu_times = np.zeros(len(ps), dtype=np.float64) if rank == 0 else None
        global_obj_values = np.zeros(len(ps), dtype=np.float64) if rank == 0 else None

        comm.Reduce(local_total_counts, global_total_counts, op=MPI.SUM, root=0)
        comm.Reduce(local_x_counts, global_x_counts, op=MPI.SUM, root=0)
        comm.Reduce(local_z_counts, global_z_counts, op=MPI.SUM, root=0)
        comm.Reduce(local_cpu_times, global_cpu_times, op=MPI.SUM, root=0)
        comm.Reduce(local_obj_values, global_obj_values, op=MPI.SUM, root=0)

        if rank == 0:
            total_ler = global_total_counts / total_shots
            x_ler = global_x_counts / total_shots
            z_ler = global_z_counts / total_shots
            
            total_ci_err = calc_ci(global_total_counts, total_shots)
            x_ci_err = calc_ci(global_x_counts, total_shots)
            z_ci_err = calc_ci(global_z_counts, total_shots)
            
            mean_obj = global_obj_values / total_shots
            avg_cpu_time = global_cpu_times / total_shots

            data_rows = []
            for j, p in enumerate(ps):
                row = {
                    'n': n, 'p': p, 'shots': total_shots,
                    'total_logical_error_rate': total_ler[j],
                    'total_err_low': total_ci_err[0][j], 'total_err_high': total_ci_err[1][j],
                    'x_logical_error_rate': x_ler[j],
                    'x_err_low': x_ci_err[0][j], 'x_err_high': x_ci_err[1][j],       
                    'z_logical_error_rate': z_ler[j],
                    'z_err_low': z_ci_err[0][j], 'z_err_high': z_ci_err[1][j],
                    'mean_objective_value': mean_obj[j], 'average_cpu_time_seconds': avg_cpu_time[j]
                }
                data_rows.append(row)
            
            new_df = pd.DataFrame(data_rows)

            if os.path.exists(filename):
                try:
                    existing_df = pd.read_csv(filename)
                    existing_df = existing_df[existing_df['n'] != n]
                    final_df = pd.concat([existing_df, new_df], ignore_index=True)
                    final_df = final_df.sort_values(by=['n', 'p'])
                except pd.errors.EmptyDataError:
                    final_df = new_df
            else:
                final_df = new_df
            final_df.to_csv(filename, index=False)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ns = [7, 17, 23, 47, 79, 103, 167]
    ps = np.logspace(-3, -1, 20).tolist()
    bias_factor = 0.5
    num_shots = 100_000

    seed = int(datetime.now().timestamp()) + rank * num_shots
    np.random.seed(seed)

    filename_base = f"hyperion_bias{bias_factor}_shots{num_shots}"
    filename = f"{filename_base}.csv"

    run_simulation(ns, ps, bias_factor, num_shots, comm, rank, size, filename)

    if rank == 0:
        print("Simulation complete. Data saved.")

if __name__ == "__main__":
    main()