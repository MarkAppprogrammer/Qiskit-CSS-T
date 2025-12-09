import galois
import numpy as np
import scipy.sparse
import scipy.stats
import sys
from datetime import datetime
from ldpc.bplsd_decoder import BpLsdDecoder
from mpi4py import MPI
import pandas as pd
import time
import os

sys.path.append('../../doubling-CSST/')
from convert_alist import readAlist

# --- Global Configuration ---
alistDirPath = "../../doubling-CSST/alistMats/GO03_self_dual/"
ALPHA = 0.05

length_dist_dict = {4:2, 6:2, 8:2, 10:2, 12:4, 14:4, 16:4, 18:4, 20:4, 22:6, 24:6, 26:6, 28:6, 30:6, 32:8, 34:6, 36:8, 38:8, 40:8, 42:8, 44:8, 46:8, 48:8, 50:8, 52:10, 54:8, 56:10, 58:10, 60:12, 62:10, 64:10}

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
        Hx, error_rate=error_rate, bp_method='product_sum', max_iter=max_iter,
        schedule='parallel', lsd_method='lsd_cs', lsd_order=lsd_order
    )
    bp_lsd_z = BpLsdDecoder(
        Hz, error_rate=error_rate, bp_method='product_sum', max_iter=max_iter,
        schedule='parallel', lsd_method='lsd_cs', lsd_order=lsd_order
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

def run_simulation(ns, ps, bias_factor, max_iter, lsd_order, total_shots, comm, rank, size, filename):
    shots_per_proc = total_shots // size
    if rank == 0:
        shots_per_proc += (total_shots % size)

    for i_n, n in enumerate(ns):
        if rank == 0:
            print(f"Simulating for n={n}...", flush=True)

        Hx = Hz = self_dual_H(n)
        Lx = Lz = np.ones((1, Hx.shape[1]), dtype=int)
        
        local_total_counts = np.zeros(len(ps), dtype=np.int64)
        local_x_counts = np.zeros(len(ps), dtype=np.int64)
        local_z_counts = np.zeros(len(ps), dtype=np.int64)
        local_cpu_times = np.zeros(len(ps), dtype=np.float64)

        for idx, error_rate in enumerate(ps):
            bp_lsd_x, bp_lsd_z = initialize_decoder(Hx, Hz, error_rate, lsd_order, max_iter)
            
            start_time = time.time()
            error_x_local, error_y_local, error_z_local = generate_errors(shots_per_proc, Hx, error_rate, bias_factor)

            for i in range(shots_per_proc):
                fail_x, fail_z = simulate_single_shot(
                    Hx, Hz, Lx, Lz, 
                    error_x_local[i], error_y_local[i], error_z_local[i], 
                    bp_lsd_x, bp_lsd_z
                )
                if fail_x: local_x_counts[idx] += 1
                if fail_z: local_z_counts[idx] += 1
                if fail_x or fail_z: local_total_counts[idx] += 1
            
            end_time = time.time()
            local_cpu_times[idx] = end_time - start_time
        
        global_total_counts = np.zeros(len(ps), dtype=np.int64) if rank == 0 else None
        global_x_counts = np.zeros(len(ps), dtype=np.int64) if rank == 0 else None
        global_z_counts = np.zeros(len(ps), dtype=np.int64) if rank == 0 else None
        global_cpu_times = np.zeros(len(ps), dtype=np.float64) if rank == 0 else None

        comm.Reduce(local_total_counts, global_total_counts, op=MPI.SUM, root=0)
        comm.Reduce(local_x_counts, global_x_counts, op=MPI.SUM, root=0)
        comm.Reduce(local_z_counts, global_z_counts, op=MPI.SUM, root=0)
        comm.Reduce(local_cpu_times, global_cpu_times, op=MPI.SUM, root=0)

        if rank == 0:
            total_ler = global_total_counts / total_shots
            x_ler = global_x_counts / total_shots
            z_ler = global_z_counts / total_shots
            avg_cpu_time = global_cpu_times / total_shots
            total_ci_err = calc_ci(global_total_counts, total_shots)
            x_ci_err = calc_ci(global_x_counts, total_shots)
            z_ci_err = calc_ci(global_z_counts, total_shots)

            data_rows = []
            for j, p in enumerate(ps):
                row = {
                    'n': n,
                    'p': p,
                    'shots': total_shots,
                    'total_logical_error_rate': total_ler[j],
                    'total_err_low': total_ci_err[0][j],
                    'total_err_high': total_ci_err[1][j],
                    'x_logical_error_rate': x_ler[j],
                    'x_err_low': x_ci_err[0][j],
                    'x_err_high': x_ci_err[1][j],
                    'z_logical_error_rate': z_ler[j],
                    'z_err_low': z_ci_err[0][j],
                    'z_err_high': z_ci_err[1][j],
                    'average_cpu_time_seconds': avg_cpu_time[j]
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
    
    seed = int(datetime.now().timestamp()) + rank * 10000
    np.random.seed(seed)

    ns = [4, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] 
    ps = np.logspace(-3, -1, 20).tolist()
    lsd_order = 20
    max_iter = 1000
    bias_factor = 0.5
    num_shots = 100_000

    filename_base = f"bplsd_bias{bias_factor}_alpha{ALPHA}_shots{num_shots}"
    filename = f"{filename_base}.csv"

    run_simulation(ns, ps, bias_factor, max_iter, lsd_order, num_shots, comm, rank, size, filename)

    if rank == 0:
        print("Simulation finished.")

if __name__ == "__main__":
    main()