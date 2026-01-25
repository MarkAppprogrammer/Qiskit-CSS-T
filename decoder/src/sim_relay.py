import numpy as np
import scipy.sparse
import scipy.stats
import sys
import time
import os
import argparse
import multiprocessing
from datetime import datetime

# Add path to finding local modules if needed, though being in the same dir helps
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from relay_bp_decoder import RelayDecoder

# Try importing MPI, else dummy it
try:
    from mpi4py import MPI
    USE_MPI = True
except ImportError:
    USE_MPI = False

# Try importing pandas
try:
    import pandas as pd
except ImportError:
    pd = None

# Try importing galois and local utils
try:
    import galois
    sys.path.append('../../doubling-CSST/')
    from convert_alist import readAlist
    HAS_GALOIS = True
except ImportError:
    HAS_GALOIS = False

# --- Global Configuration ---
# Match sim_lsd.py logic where possible
alistDirPath = "../../doubling-CSST/alistMats/GO03_self_dual/"
ALPHA = 0.05

length_dist_dict = {4:2, 6:2, 8:2, 10:2, 12:4, 14:4, 16:4, 18:4, 20:4, 22:6, 24:6, 26:6, 28:6, 30:6, 32:8, 34:6, 36:8, 38:8, 40:8, 42:8, 44:8, 46:8, 48:8, 50:8, 52:10, 54:8, 56:10, 58:10, 60:12, 62:10, 64:10}

def self_dual_H(n):
    if HAS_GALOIS:
        d = length_dist_dict.get(n, n//2) # Fallback if n not in dict
        F2 = galois.GF(2)
        alistFilePath = alistDirPath + "n" + str(n) + "_d" + str(d) + ".alist"
        if os.path.exists(alistFilePath):
            GenMat = F2(readAlist(alistFilePath))
            G_punctured = GenMat[:, :-1]
            H_punctured = G_punctured.null_space()
            H = scipy.sparse.csr_matrix(np.array(H_punctured, dtype=int))
            return H
    
    # Fallback/Dummy if galois not present or file missing
    # Create a simple repetition code-like matrix or random CSS for testing
    # This ensures the script runs even without external data
    print(f"Warning: Using dummy H matrix for n={n}")
    return scipy.sparse.csr_matrix(np.eye(n, k=1) + np.eye(n), dtype=int)

def initialize_decoder(Hx, Hz, error_rate, **kwargs):
    # kwargs can contain pre_iter, num_legs, etc.
    pre_iter = kwargs.get('pre_iter', 20)
    num_legs = kwargs.get('num_legs', 10)
    
    priors_x = np.full(Hx.shape[1], error_rate)
    decoder_x = RelayDecoder(Hx, priors_x, pre_iter=pre_iter, num_legs=num_legs)
    
    priors_z = np.full(Hz.shape[1], error_rate)
    decoder_z = RelayDecoder(Hz, priors_z, pre_iter=pre_iter, num_legs=num_legs)
    
    return decoder_x, decoder_z

def generate_errors(num_shots, H, error_probability, bias_factor):
    rZ = bias_factor / (1 + bias_factor)
    rX = rY = (1 - rZ) / 2
    
    n = H.shape[1]
    error_x = np.random.choice([0, 1], size=(num_shots, n), p=[1 - rX * error_probability, rX * error_probability])
    error_y = np.random.choice([0, 1], size=(num_shots, n), p=[1 - rY * error_probability, rY * error_probability])
    error_z = np.random.choice([0, 1], size=(num_shots, n), p=[1 - rZ * error_probability, rZ * error_probability])
    return error_x, error_y, error_z

def simulate_single_shot(Hx, Hz, Lx, Lz, error_x, error_y, error_z, decoder_x, decoder_z):
    # Combine errors
    real_error_x = (error_x + error_y) % 2
    real_error_z = (error_z + error_y) % 2
    
    # Syndromes
    # Z errors cause X syndrome (checked by Hx)
    # X errors cause Z syndrome (checked by Hz)
    syndromes_x = (real_error_z @ Hx.T) % 2
    syndromes_z = (real_error_x @ Hz.T) % 2
    
    # Decode
    # decoder_x decodes Z errors using Hx
    success_z, correction_z = decoder_x.decode(syndromes_x)
    
    # decoder_z decodes X errors using Hz
    success_x, correction_x = decoder_z.decode(syndromes_z)
    
    # Residuals
    residual_error_z = (correction_z + real_error_z) % 2
    residual_error_x = (correction_x + real_error_x) % 2
    
    # Logical Failure Check
    logical_fail_x = (residual_error_z @ Lx.T) % 2
    logical_fail_z = (residual_error_x @ Lz.T) % 2
    
    return int(np.any(logical_fail_x)), int(np.any(logical_fail_z))

# Worker function for multiprocessing
def worker_task(task_args):
    # Unpack arguments
    Hx, Hz, Lx, Lz, error_rate, bias_factor, num_legs, shots, seed = task_args
    
    # Set seed for this worker
    np.random.seed(seed)
    
    # Initialize decoders ONCE per worker task
    decoder_x, decoder_z = initialize_decoder(Hx, Hz, error_rate, num_legs=num_legs)
    
    # Generate ALL errors in batch (vectorized)
    error_x_batch, error_y_batch, error_z_batch = generate_errors(shots, Hx, error_rate, bias_factor)
    
    local_x = 0
    local_z = 0
    local_total = 0
    
    # Loop over shots
    for i in range(shots):
        fail_x, fail_z = simulate_single_shot(
            Hx, Hz, Lx, Lz, 
            error_x_batch[i], error_y_batch[i], error_z_batch[i], 
            decoder_x, decoder_z
        )
        if fail_x: local_x += 1
        if fail_z: local_z += 1
        if fail_x or fail_z: local_total += 1
        
    return local_total, local_x, local_z

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

def run_simulation(ns, ps, bias_factor, num_legs, total_shots, filename):
    # Correct Path Construction relative to script location
    # Target: ../../fig/RelayBP/filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "../../fig/RelayBP/")
    full_path = os.path.join(target_dir, filename)
    
    # Ensure directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Determine parallelism
    num_workers = multiprocessing.cpu_count()
    if num_workers > 1:
        pool = multiprocessing.Pool(processes=num_workers)
        shots_per_worker = total_shots // num_workers
        remainder = total_shots % num_workers
        # Distribute shots: first 'remainder' workers get +1 shot
        shots_dist = [shots_per_worker + 1 if i < remainder else shots_per_worker for i in range(num_workers)]
    else:
        num_workers = 1
        shots_dist = [total_shots]

    base_seed = int(datetime.now().timestamp())

    for i_n, n in enumerate(ns):
        print(f"Simulating for n={n}...", flush=True)

        Hx = self_dual_H(n)
        Hz = Hx # Self-dual code
        Lx = Lz = np.ones((1, Hx.shape[1]), dtype=int)
        
        local_total_counts = np.zeros(len(ps), dtype=np.int64)
        local_x_counts = np.zeros(len(ps), dtype=np.int64)
        local_z_counts = np.zeros(len(ps), dtype=np.int64)
        local_cpu_times = np.zeros(len(ps), dtype=np.float64)

        for idx, error_rate in enumerate(ps):
            start_time = time.time()
            
            if num_workers > 1:
                # Prepare tasks
                tasks = []
                for i in range(num_workers):
                    if shots_dist[i] > 0:
                        worker_seed = base_seed + i * 1000 + idx
                        tasks.append((Hx, Hz, Lx, Lz, error_rate, bias_factor, num_legs, shots_dist[i], worker_seed))
                
                results = pool.map(worker_task, tasks)
                
                # Aggregate results
                for res in results:
                    local_total_counts[idx] += res[0]
                    local_x_counts[idx] += res[1]
                    local_z_counts[idx] += res[2]
            else:
                # Serial execution fallback
                task_args = (Hx, Hz, Lx, Lz, error_rate, bias_factor, num_legs, total_shots, base_seed)
                res = worker_task(task_args)
                local_total_counts[idx] = res[0]
                local_x_counts[idx] = res[1]
                local_z_counts[idx] = res[2]

            end_time = time.time()
            local_cpu_times[idx] = end_time - start_time
        
        # Save results (Single process logic, ignoring MPI for simplicity as requested)
        if pd is not None:
            total_ler = local_total_counts / total_shots
            x_ler = local_x_counts / total_shots
            z_ler = local_z_counts / total_shots
            avg_cpu_time = local_cpu_times / total_shots
            total_ci_err = calc_ci(local_total_counts, total_shots)
            x_ci_err = calc_ci(local_x_counts, total_shots)
            z_ci_err = calc_ci(local_z_counts, total_shots)

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
            if os.path.exists(full_path):
                try:
                    existing_df = pd.read_csv(full_path)
                    existing_df = existing_df[existing_df['n'] != n]
                    final_df = pd.concat([existing_df, new_df], ignore_index=True)
                    final_df = final_df.sort_values(by=['n', 'p'])
                except:
                    final_df = new_df
            else:
                final_df = new_df
            final_df.to_csv(full_path, index=False)
            print(f"Data saved to {full_path}")

    if num_workers > 1:
        pool.close()
        pool.join()

def main():
    parser = argparse.ArgumentParser(description="Relay-BP Simulation")
    parser.add_argument("--legs", type=int, default=10, help="Number of relay legs")
    args = parser.parse_args()

    # Params
    ns = [4, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    ps = np.logspace(-3, -1, 20).tolist()
    
    shot_counts = [10, 100, 1000, 10000, 100000, 1000000]
    biases = [0.0, 0.5]
    
    for shots in shot_counts:
        for bias_factor in biases:
            print(f"--- Running simulation for {shots} shots, bias {bias_factor} ---")
            filename = f"relaybp_bias{bias_factor}_shots{shots}.csv"
            run_simulation(ns, ps, bias_factor, args.legs, shots, filename)

if __name__ == "__main__":
    main()
