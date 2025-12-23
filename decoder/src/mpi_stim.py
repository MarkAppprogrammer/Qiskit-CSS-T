import sys
import os
import tempfile
import argparse
import numpy as np
import pandas as pd
import galois
import stim
import sinter
from typing import List, Tuple, Dict

sys.path.append(".")

if not os.path.exists("convert_alist.py"):
    sys.path.append('../../doubling-CSST/')

try:
    from convert_alist import readAlist
    from mwpf.sinter_decoders import SinterMWPFDecoder
    from noise_models import (
        standard_depolarizing_noise_model, 
        si1000_noise_model, 
        bravyi_noise_model
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import dependencies. {e}")
    sys.exit(1)

# --- Path Configuration ---
LOCAL_ALIST_DIR = "./alistMats/"
DEV_ALIST_DIR = "../../doubling-CSST/alistMats/QR_dual_containing/"

if os.path.exists(LOCAL_ALIST_DIR):
    ALIST_DIR_PATH = LOCAL_ALIST_DIR
else:
    ALIST_DIR_PATH = DEV_ALIST_DIR

LENGTH_DIST_DICT = {7:3, 17:5, 23:7, 47:11, 79:15, 103:19, 167:23} 

# --- Helper Functions ---

def get_parity_matrices(n: int, if_self_dual: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if n not in LENGTH_DIST_DICT:
        raise ValueError(f"No distance defined for n={n}")
    
    d = LENGTH_DIST_DICT[n]
    alist_file_path = os.path.join(ALIST_DIR_PATH, f"n{n}_d{d}.alist")
    
    if not os.path.exists(alist_file_path):
        raise FileNotFoundError(f"Alist file not found: {alist_file_path}")

    F2 = galois.GF(2)
    GenMat = F2(readAlist(alist_file_path))
    H_punctured = GenMat[:, :-1].null_space() if if_self_dual else GenMat.null_space()
    return np.array(H_punctured, dtype=np.uint8), np.array(H_punctured, dtype=np.uint8)

def check_css_validity(Hx: np.ndarray, Hz: np.ndarray):
    if np.any((Hx @ Hz.T) % 2):
        print(f"CRITICAL WARNING: Hx and Hz do NOT commute!")

def find_logical_z_operator(Hx: np.ndarray, Hz: np.ndarray) -> np.ndarray:
    num_qubits = Hx.shape[1]
    candidates = [np.ones(num_qubits, dtype=np.uint8)]
    for _ in range(100):
        candidates.append(np.random.randint(0, 2, size=num_qubits).astype(np.uint8))

    def gf2_rank(rows):
        rows = np.array(rows, copy=True)
        pivot_row = 0
        num_rows, num_cols = rows.shape
        for col in range(num_cols):
            if pivot_row >= num_rows: break
            if rows[pivot_row, col] == 0:
                candidates_below = rows[pivot_row:, col]
                if not np.any(candidates_below): continue
                swap = np.argmax(candidates_below) + pivot_row
                rows[[pivot_row, swap]] = rows[[swap, pivot_row]]
            for r in range(pivot_row + 1, num_rows):
                if rows[r, col]: rows[r] ^= rows[pivot_row]
            pivot_row += 1
        return pivot_row

    base_rank = gf2_rank(Hz)
    for candidate in candidates:
        if np.any((Hx @ candidate) % 2): continue
        m_stack = np.vstack([Hz, candidate])
        if gf2_rank(m_stack) > base_rank:
            return candidate

    print("WARNING: Could not find logical Z. Using placeholder.")
    return np.zeros(num_qubits, dtype=np.uint8)

# --- Circuit Generation ---

def generate_css_memory_experiment(Hx: np.ndarray, Hz: np.ndarray, rounds: int) -> stim.Circuit:
    check_css_validity(Hx, Hz)
    num_data = Hx.shape[1]
    num_x_checks = Hx.shape[0]
    num_z_checks = Hz.shape[0]
    
    data_qubits = list(range(num_data))
    x_ancillas = list(range(num_data, num_data + num_x_checks))
    z_ancillas = list(range(num_data + num_x_checks, num_data + num_x_checks + num_z_checks))
    
    circuit = stim.Circuit()
    circuit.append("R", data_qubits + x_ancillas + z_ancillas)
    circuit.append("TICK")
    
    def append_measurement_round(c: stim.Circuit, is_first_round: bool):
        c.append("H", x_ancillas)
        for check_idx, row in enumerate(Hx):
            targets = [data_qubits[q] for q in np.flatnonzero(row)]
            ancilla = x_ancillas[check_idx]
            for t in targets:
                c.append("CX", [ancilla, t])
        c.append("H", x_ancillas)
        c.append("M", x_ancillas)
        
        for check_idx, row in enumerate(Hz):
            targets = [data_qubits[q] for q in np.flatnonzero(row)]
            ancilla = z_ancillas[check_idx]
            for t in targets:
                c.append("CX", [t, ancilla])
        c.append("M", z_ancillas)
        c.append("R", x_ancillas + z_ancillas)
        c.append("TICK")

        total_measurements_per_round = num_x_checks + num_z_checks
        
        for i in range(num_x_checks):
            current_rec = stim.target_rec(-total_measurements_per_round + i)
            if not is_first_round:
                prev_rec = stim.target_rec(-total_measurements_per_round * 2 + i)
                c.append("DETECTOR", [current_rec, prev_rec], [i, 0, 0])

        for i in range(num_z_checks):
            offset_in_round = num_x_checks + i
            current_rec = stim.target_rec(-total_measurements_per_round + offset_in_round)
            if not is_first_round:
                prev_rec = stim.target_rec(-total_measurements_per_round * 2 + offset_in_round)
                c.append("DETECTOR", [current_rec, prev_rec], [num_x_checks + i, 0, 0])
            else:
                c.append("DETECTOR", [current_rec], [num_x_checks + i, 0, 0])

    loop_body = stim.Circuit()
    append_measurement_round(loop_body, is_first_round=False)
    append_measurement_round(circuit, is_first_round=True)
    
    if rounds > 1:
        circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop_body))

    circuit.append("M", data_qubits)
    for i, row in enumerate(Hz):
        rec_targets = []
        data_indices = np.flatnonzero(row)
        for data_idx in data_indices:
            offset = -(num_data - data_idx)
            rec_targets.append(stim.target_rec(offset))
        z_ancilla_offset = -(num_data + num_z_checks - i)
        rec_targets.append(stim.target_rec(z_ancilla_offset))
        circuit.append("DETECTOR", rec_targets, [num_x_checks + i, 1, 0])

    L_Z_bits = find_logical_z_operator(Hx, Hz)
    obs_targets = []
    for k in np.flatnonzero(L_Z_bits):
        offset = -(num_data - k)
        obs_targets.append(stim.target_rec(offset))
    circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)
    
    return circuit

def generate_experiment_with_noise(Hx: np.ndarray, Hz: np.ndarray, rounds: int, 
                                   noise_model_name: str, noise_params: Dict[str, float]) -> stim.Circuit:
    
    clean_circuit = generate_css_memory_experiment(Hx, Hz, rounds)
    num_data = Hx.shape[1]
    num_check = Hx.shape[0] + Hz.shape[0]
    data_qubits = list(range(num_data))
    all_qubits = list(range(num_data + num_check))
    
    base_p = noise_params.get('p', 0.001) if noise_params else 0.001

    if noise_model_name == "depolarizing":
        return standard_depolarizing_noise_model(
            circuit=clean_circuit,
            data_qubits=data_qubits,
            after_clifford_depolarization=base_p,
            after_reset_flip_probability=base_p,
            before_measure_flip_probability=base_p,
            before_round_data_depolarization=base_p
        )
    elif noise_model_name == "si1000":
        return si1000_noise_model(
            circuit=clean_circuit, 
            data_qubits=data_qubits, 
            all_qubits=all_qubits,
            probability=base_p
        )
    elif noise_model_name == "bravyi":
        return bravyi_noise_model(circuit=clean_circuit, error_rate=base_p)
    else:
        raise ValueError(f"Unknown noise model: {noise_model_name}")

# --- Data Processing ---

def parse_and_average_stats(stats: List[sinter.TaskStats], trace_file: str, model_name: str) -> pd.DataFrame:
    try:
        df_details = SinterMWPFDecoder.parse_mwpf_trace(trace_file)
        avg_obj = df_details['objective_value'].mean()
        avg_cpu = df_details['cpu_time'].mean()
    except Exception:
        avg_obj = 0.0
        avg_cpu = 0.0
    
    results = []
    for s in stats:
        m = s.json_metadata
        logical_err = s.errors / s.shots if s.shots > 0 else 0
        results.append({
            'noise_model': model_name,
            'n': m['n'], 'd': m['d'], 'r': m['r'], 'p': m['p'],
            'shots': s.shots, 'errors': s.errors,
            'total_logical_error_rate': logical_err,
            'mean_objective_value': avg_obj,
            'average_cpu_time_seconds': avg_cpu,
        })
    return pd.DataFrame(results)

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    default_workers = int(slurm_cpus) if slurm_cpus else os.cpu_count()

    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--max_shots", type=int, default=100_000)
    parser.add_argument("--output", type=str, default="hyperion_results.csv")
    args = parser.parse_args()

    # Simulation Parameters
    noise_values = [0.008, 0.009, 0.01, 0.011, 0.012]
    n_list = [7, 23, 47]
    
    # List of models to run sequentially
    models_to_run = ["depolarizing", "si1000"]

    print(f"Running with {args.workers} workers.")
    print(f"Using Alist path: {ALIST_DIR_PATH}")

    # Prepare base filename for splitting
    # e.g., "results_v1234.csv" -> base="results_v1234", ext=".csv"
    output_base, output_ext = os.path.splitext(args.output)

    for model_name in models_to_run:
        print(f"\n=== Generating tasks for model: {model_name} ===")
        tasks = []
        
        for n in n_list:
            try:
                d = LENGTH_DIST_DICT[n]
                Hx, Hz = get_parity_matrices(n, if_self_dual=True)
                num_rounds = d * 3

                for p in noise_values:
                    circuit = generate_experiment_with_noise(
                        Hx=Hx, Hz=Hz, rounds=num_rounds, 
                        noise_model_name=model_name, 
                        noise_params={"p": p}
                    )
                    tasks.append(sinter.Task(
                        circuit=circuit,
                        decoder='mwpf',
                        json_metadata={'n': n, 'd': d, 'r': num_rounds, 'p': p}
                    ))
            except Exception as e:
                print(f"Skipping n={n}: {e}")

        if not tasks:
            print(f"No tasks generated for {model_name}. Skipping.")
            continue

        # Create unique temp trace file for this batch
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            trace_path = tmp.name
        
        try:
            print(f"Initializing SinterMWPFDecoder (trace: {trace_path})")
            mwpf_decoder = SinterMWPFDecoder(cluster_node_limit=15, trace_filename=trace_path)

            print(f"Collecting stats for {model_name}...")
            collected_stats = sinter.collect(
                num_workers=args.workers,
                tasks=tasks,
                custom_decoders={"mwpf": mwpf_decoder},
                max_shots=args.max_shots,
                print_progress=True,
            )

            final_df = parse_and_average_stats(collected_stats, trace_path, model_name)
            
            # Save separate CSV
            specific_output_filename = f"{output_base}_{model_name}{output_ext}"
            final_df.to_csv(specific_output_filename, index=False)
            print(f"Saved results to: {specific_output_filename}")

        finally:
            if os.path.exists(trace_path):
                os.remove(trace_path)

if __name__ == "__main__":
    main()