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

# Add paths
sys.path.append('../../doubling-CSST/')

# Imports
try:
    from convert_alist import readAlist
    from mwpf.sinter_decoders import SinterMWPFDecoder
    # Import the refactored noise models
    from noise_models import (
        standard_depolarizing_noise_model, 
        si1000_noise_model, 
        bravyi_noise_model
    )
except ImportError as e:
    print(f"Import Error: {e}. Check paths.")
    sys.exit(1)

ALIST_DIR_PATH = "../../doubling-CSST/alistMats/QR_dual_containing/" 
LENGTH_DIST_DICT = {7:3, 17:5, 23:7, 47:11, 79:15, 103:19, 167:23} 

# --- Section 1: Mathematical & Matrix Functions ---

def get_parity_matrices(n: int, if_self_dual: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if n not in LENGTH_DIST_DICT:
        raise ValueError(f"No distance defined for n={n}")
    
    d = LENGTH_DIST_DICT[n]
    alist_file_path = f"{ALIST_DIR_PATH}n{n}_d{d}.alist"
    
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

# --- Section 2: Circuit Generation ---

def generate_css_memory_experiment(Hx: np.ndarray, Hz: np.ndarray, rounds: int) -> stim.Circuit:
    """
    Generates a CLEAN circuit (no noise). Noise is applied later by the noise models.
    Added 'TICK' instructions to support time-step based noise models.
    """
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
        c.append("M", x_ancillas) # Note: Removed explicit noise_p here, handled by noise model
        
        for check_idx, row in enumerate(Hz):
            targets = [data_qubits[q] for q in np.flatnonzero(row)]
            ancilla = z_ancillas[check_idx]
            for t in targets:
                c.append("CX", [t, ancilla])
        c.append("M", z_ancillas)
        c.append("R", x_ancillas + z_ancillas)
        c.append("TICK") # Crucial for noise models to identify rounds

        total_measurements_per_round = num_x_checks + num_z_checks
        
        # Detectors
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
    
    # 1. Generate Clean Circuit
    clean_circuit = generate_css_memory_experiment(Hx, Hz, rounds)
    
    # 2. Extract Qubit Info for Noise Models
    num_data = Hx.shape[1]
    num_check = Hx.shape[0] + Hz.shape[0]
    data_qubits = list(range(num_data))
    all_qubits = list(range(num_data + num_check)) # Assuming compact indices
    
    base_p = noise_params.get('p', 0.001) if noise_params else 0.001

    # 3. Apply Noise Models
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
            all_qubits=all_qubits, # Passed for idle noise calculations
            probability=base_p
        )
    elif noise_model_name == "bravyi":
        return bravyi_noise_model(circuit=clean_circuit, error_rate=base_p)
    else:
        raise ValueError(f"Unknown noise model: {noise_model_name}")

# --- Section 3: Data Processing ---

def parse_and_average_stats(stats: List[sinter.TaskStats], trace_file: str) -> pd.DataFrame:
    print("Parsing detailed MWPF trace...")
    try:
        df_details = SinterMWPFDecoder.parse_mwpf_trace(trace_file)
        avg_obj = df_details['objective_value'].mean()
        avg_cpu = df_details['cpu_time'].mean()
    except Exception:
        print("Warning: Could not parse MWPF trace. Using default values.")
        avg_obj = 0.0
        avg_cpu = 0.0
    
    results = []
    for s in stats:
        m = s.json_metadata
        logical_err = s.errors / s.shots if s.shots > 0 else 0
        results.append({
            'n': m['n'], 'd': m['d'], 'r': m['r'], 'p': m['p'],
            'shots': s.shots, 'errors': s.errors,
            'total_logical_error_rate': logical_err,
            'mean_objective_value': avg_obj,
            'average_cpu_time_seconds': avg_cpu,
        })
    return pd.DataFrame(results)

# --- Section 4: Main Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--max_shots", type=int, default=100_000)
    parser.add_argument("--output", type=str, default="hyperion_results.csv")
    args = parser.parse_args()

    # Parameters
    noise_values = [0.008, 0.009, 0.01, 0.011, 0.012]
    n_list = [7] 

    tasks = []
    print(f"Generating tasks for noise model: si1000")
    
    for n in n_list:
        try:
            d = LENGTH_DIST_DICT[n]
            Hx, Hz = get_parity_matrices(n, if_self_dual=True)
            num_rounds = d * 3  # Requirement: rounds = 3 * d

            for p in noise_values:
                # Calls the wrapper which calls the refactored noise model
                circuit = generate_experiment_with_noise(
                    Hx=Hx, Hz=Hz, rounds=num_rounds, 
                    noise_model_name="si1000", 
                    noise_params={"p": p}
                )
                
                tasks.append(
                    sinter.Task(
                        circuit=circuit,
                        decoder='mwpf',
                        json_metadata={'n': n, 'd': d, 'r': num_rounds, 'p': p}
                    )
                )
        except Exception as e:
            print(f"Skipping n={n}: {e}")

    if not tasks:
        print("No tasks generated. Exiting.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        trace_path = tmp.name
    
    try:
        print(f"Initializing SinterMWPFDecoder with trace: {trace_path}")
        mwpf_decoder = SinterMWPFDecoder(cluster_node_limit=15, trace_filename=trace_path)

        print(f"Starting collection on {args.workers} workers...")
        collected_stats = sinter.collect(
            num_workers=args.workers,
            tasks=tasks,
            custom_decoders={"mwpf": mwpf_decoder},
            max_shots=args.max_shots,
            print_progress=True,
        )

        final_df = parse_and_average_stats(collected_stats, trace_path)
        final_df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

    finally:
        if os.path.exists(trace_path):
            os.remove(trace_path)

if __name__ == "__main__":
    main()