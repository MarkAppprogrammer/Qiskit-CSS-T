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

# --- Configuration Constants ---

# Paths relative to the script location or dev location
BASE_ALIST_DIR_DEV = "../../doubling-CSST/alistMats/"
BASE_ALIST_DIR_LOCAL = "./alistMats/"

# Select base path
if os.path.exists(BASE_ALIST_DIR_LOCAL):
    BASE_ALIST_PATH = BASE_ALIST_DIR_LOCAL
else:
    BASE_ALIST_PATH = BASE_ALIST_DIR_DEV

# Option 1: GO03 Self Dual Parameters
OP1_DIR = os.path.join(BASE_ALIST_PATH, "GO03_self_dual/")
OP1_DICT = {4:2, 6:2, 8:2, 10:2, 12:4, 14:4, 16:4, 18:4, 20:4, 22:6, 24:6, 26:6, 28:6, 30:6, 32:8, 34:6, 36:8, 38:8, 40:8, 42:8, 44:8, 46:8, 48:8, 50:8, 52:10, 54:8, 56:10, 58:10, 60:12, 62:10, 64:10}

# Option 2: QR Dual Containing Parameters
OP2_DIR = os.path.join(BASE_ALIST_PATH, "QR_dual_containing/")
OP2_DICT = {7:3, 17:5, 23:7, 47:11, 79:15, 103:19, 167:23}

# Master Configuration Dictionary
CODE_CONFIGS = {
    "self_dual": {
        "dir": OP1_DIR,
        "dist_dict": OP1_DICT,
        "if_self_dual": True, # Triggers puncturing logic
        "default_n_list": [8, 12, 16] # Example defaults
    },
    "dual_containing": {
        "dir": OP2_DIR,
        "dist_dict": OP2_DICT,
        "if_self_dual": False, # Triggers direct null space logic
        "default_n_list": [7, 23, 47] # Example defaults
    }
}

# --- Helper Functions ---

def get_parity_matrices(n: int, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves Hx and Hz based on the configuration dictionary.
    """
    dist_dict = config["dist_dict"]
    alist_dir = config["dir"]
    is_self_dual_logic = config["if_self_dual"]

    if n not in dist_dict:
        raise ValueError(f"No distance defined for n={n} in selected code type.")
    
    d = dist_dict[n]
    alist_file_path = os.path.join(alist_dir, f"n{n}_d{d}.alist")
    
    if not os.path.exists(alist_file_path):
        raise FileNotFoundError(f"Alist file not found: {alist_file_path}")

    F2 = galois.GF(2)
    GenMat = F2(readAlist(alist_file_path))

    # --- Logic Switch based on Option 1 vs Option 2 ---
    if is_self_dual_logic:
        # Option 1: Puncture last column
        # G_punctured = GenMat[:, :-1]
        H_punctured = GenMat[:, :-1].null_space()
    else:
        # Option 2: Direct Null Space
        H_punctured = GenMat.null_space()
    
    # Cast to standard numpy uint8 for Stim
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
        if not df_details.empty:
            avg_obj = df_details['objective_value'].mean()
            avg_cpu = df_details['cpu_time'].mean()
        else:
            avg_obj = 0.0
            avg_cpu = 0.0
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
            'code_type': m.get('code_type', 'unknown'),
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
    parser.add_argument("--max_shots", type=int, default=1_000_000)
    parser.add_argument("--output", type=str, default="hyperion_results.csv")
    parser.add_argument("--code_type", type=str, choices=["self_dual", "dual_containing"], required=True,
                        help="Choose 'self_dual' (Option 1) or 'dual_containing' (Option 2)")
    
    args = parser.parse_args()

    # --- SHARDING CONFIGURATION ---
    proc_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    if world_size > 1:
        print(f"[Node {proc_rank}/{world_size}] Sharding enabled. Using {args.workers} cores on this node.")

    # --- SETUP BASED ON ARGUMENT ---
    selected_config = CODE_CONFIGS[args.code_type]
    if args.code_type == "self_dual":
        n_list = [8, 12, 16]
    else:
        n_list = [7, 23, 47]

    noise_values = [0.008, 0.009, 0.01, 0.011, 0.012]
    models_to_run = ["depolarizing", "si1000"]
    
    print(f"[Node {proc_rank}] Running Code Type: {args.code_type}")
    print(f"[Node {proc_rank}] Alist Directory: {selected_config['dir']}")
    print(f"[Node {proc_rank}] Logic: {'Punctured' if selected_config['if_self_dual'] else 'Direct Null Space'}")

    output_base, output_ext = os.path.splitext(args.output)

    # We loop through models, but we collect ALL tasks first to shard them fairly
    for model_name in models_to_run:
        all_tasks = []
        
        # 1. Generate ALL tasks for this model
        for n in n_list:
            try:
                # Use the config to get matrices
                Hx, Hz = get_parity_matrices(n, selected_config)
                d = selected_config['dist_dict'][n]
                num_rounds = d * 3

                for p in noise_values:
                    circuit = generate_experiment_with_noise(
                        Hx=Hx, Hz=Hz, rounds=num_rounds, 
                        noise_model_name=model_name, 
                        noise_params={"p": p}
                    )
                    
                    task = sinter.Task(
                        circuit=circuit,
                        decoder='mwpf',
                        json_metadata={
                            'n': n, 'd': d, 'r': num_rounds, 'p': p, 
                            'noise_model': model_name, 
                            'code_type': args.code_type
                        }
                    )
                    all_tasks.append(task)
            except Exception as e:
                print(f"[Node {proc_rank}] Skipping n={n}: {e}")

        # 2. Filter tasks: Keep only the ones assigned to THIS node
        my_tasks = [t for i, t in enumerate(all_tasks) if i % world_size == proc_rank]

        if not my_tasks:
            print(f"[Node {proc_rank}] No tasks assigned for {model_name}. Skipping.")
            continue

        print(f"[Node {proc_rank}] Processing {len(my_tasks)} out of {len(all_tasks)} tasks for {model_name}...")

        # 3. Run Sinter on this node's slice of work
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            trace_path = tmp.name
            
        resume_file = f"{output_base}_{model_name}_rank{proc_rank}_partial.csv"
        
        try:
            mwpf_decoder = SinterMWPFDecoder(cluster_node_limit=200, trace_filename=trace_path)
            
            collected_stats = sinter.collect(
                num_workers=args.workers,
                tasks=my_tasks,
                custom_decoders={"mwpf": mwpf_decoder},
                max_shots=args.max_shots,
                print_progress=False,
                save_resume_filepath=resume_file
            )

            final_df = parse_and_average_stats(collected_stats, trace_path, model_name)
            part_filename = f"{output_base}_{model_name}_rank{proc_rank}{output_ext}"
            final_df.to_csv(part_filename, index=False)
            print(f"[Node {proc_rank}] Saved partial results to: {part_filename}")

        finally:
            if os.path.exists(trace_path):
                os.remove(trace_path)

if __name__ == "__main__":
    main()