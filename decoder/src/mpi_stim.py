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
        "if_self_dual": True,
        "default_n_list": [8, 18, 24]
    },
    "dual_containing": {
        "dir": OP2_DIR,
        "dist_dict": OP2_DICT,
        "if_self_dual": False,
        "default_n_list": [7, 17, 23]
    }
}

# --- Helper Functions ---
def H(n, if_self_dual=False):
    d = length_dist_dict[n]
    F2 = galois.GF(2)
    if n == 17:
        alistFilePath = alistDirPath + "n" + str(n) + "_d" + str(d) + "_Hz.alist"
        Hz = F2(readAlist(alistFilePath))
        alistFilePath = alistDirPath + "n" + str(n) + "_d" + str(d) + "_Hx.alist"
        Hx = F2(readAlist(alistFilePath))
    else:
        alistFilePath = alistDirPath + "n" + str(n) + "_d" + str(d) + alistEnd

    GenMat = F2(readAlist(alistFilePath))
    
    if if_self_dual:
        G_punctured = GenMat[:, :-1]  # Puncture the last column
        Hz = Hx = G_punctured.null_space() # Parity-check matrix of the punctured code = generator matrix of the dual of the punctured code
    else:
        if n != 17:
            Hz = Hx = GenMat.null_space()  # For GenMat being a dual-containing code, directly use the null-space of the generator matrix directly as the parity-check matrix

    Hx = np.array(Hx, dtype=np.uint8)
    Hz = np.array(Hz, dtype=np.uint8)

    return Hx, Hz

def get_logical_operators(Hx, Hz):
    """
    Finds one pair of Logical X and Logical Z operators for a CSS code.
    Assumes one logical qubit for simplicity (k=2).
    """
    GF = galois.GF(2)
    gf_Hx = GF(Hx.astype(int))
    gf_Hz = GF(Hz.astype(int))

    lz_candidates = gf_Hx.null_space()
    lx_candidates = gf_Hz.null_space()

    L_Z = None
    L_X = None

    L_Z = np.array(lz_candidates[0], dtype=np.uint8) 
    L_X = np.array(lx_candidates[0], dtype=np.uint8)
    
    return L_X, L_Z

def get_start_x(num_items: int, spacing: float, center_point: float) -> float:
    """Calculates the starting X coordinate to center a row of items."""
    if num_items <= 1: 
        return center_point
    total_width = (num_items - 1) * spacing
    return center_point - (total_width / 2.0)

def append_layered_cnots(c: stim.Circuit, check_matrix, ancilla_idxs, data_idxs, ancilla_is_control):
    check_targets = []
    max_weight = 0
    for row in check_matrix:
        targets = [data_idxs[q] for q in np.flatnonzero(row)]
        check_targets.append(targets)
        max_weight = max(max_weight, len(targets))
    
    for k in range(max_weight):
        layer_args = []
        for i, targets in enumerate(check_targets):
            if k < len(targets):
                anc = ancilla_idxs[i]
                dat = targets[k]
                if ancilla_is_control:
                    layer_args.extend([anc, dat])
                else:
                    layer_args.extend([dat, anc])
        if layer_args:
            c.append("CX", layer_args)
            c.append("TICK")

def generate_css_memory_experiment(
    Hx: np.ndarray, 
    Hz: np.ndarray, 
    rounds: int,
    memory_basis: str = "Z", # "X" or "Z"
) -> stim.Circuit:
    
    num_data = Hx.shape[1]
    num_x_checks = Hx.shape[0]
    num_z_checks = Hz.shape[0]
    
    data_qubits = list(range(num_data))
    x_ancillas = list(range(num_data, num_data + num_x_checks))
    z_ancillas = list(range(num_data + num_x_checks, num_data + num_x_checks + num_z_checks))
    
    circuit = stim.Circuit()

    # --- 1. Layout / Coordinates ---
    v_spacing = 2.0 
    data_h_spacing = 2.0
    ancilla_h_spacing = 4.0 

    data_width = (num_data - 1) * data_h_spacing
    data_center_x = data_width / 2.0

    # Store coordinates for reuse in detectors
    x_anc_coords = []
    z_anc_coords = []

    # Row 0: X-Ancillas (Top)
    x_start_x = get_start_x(num_x_checks, ancilla_h_spacing, data_center_x)
    for i, q in enumerate(x_ancillas):
        pos = [x_start_x + i * ancilla_h_spacing, 0]
        circuit.append("QUBIT_COORDS", [q], pos)
        x_anc_coords.append(pos)

    # Row 1: Data Qubits (Middle)
    for i, q in enumerate(data_qubits):
        circuit.append("QUBIT_COORDS", [q], [i * data_h_spacing, v_spacing])

    # Row 2: Z-Ancillas (Bottom)
    z_start_x = get_start_x(num_z_checks, ancilla_h_spacing, data_center_x)
    for i, q in enumerate(z_ancillas):
        pos = [z_start_x + i * ancilla_h_spacing, v_spacing * 2]
        circuit.append("QUBIT_COORDS", [q], pos)
        z_anc_coords.append(pos)

    # --- 2. Initialization ---
    if memory_basis == "Z":
        circuit.append("R", data_qubits) 
    elif memory_basis == "X":
        circuit.append("RX", data_qubits) 
    
    circuit.append("R", x_ancillas + z_ancillas)
    circuit.append("TICK")

    # --- 3. Round Function ---
    def append_measurement_round(c: stim.Circuit, is_first_round: bool):
        # === X-Stabilizers ===
        c.append("H", x_ancillas)
        c.append("TICK")
        append_layered_cnots(c, Hx, x_ancillas, data_qubits, ancilla_is_control=True)
        c.append("H", x_ancillas)
        c.append("M", x_ancillas)
        c.append("TICK")
        
        # === Z-Stabilizers ===
        append_layered_cnots(c, Hz, z_ancillas, data_qubits, ancilla_is_control=False)
        c.append("M", z_ancillas)
        c.append("R", x_ancillas + z_ancillas)

        # === Detectors ===
        total_meas_per_round = num_x_checks + num_z_checks
        
        # X-Check Detectors
        for i in range(num_x_checks):
            current_rec = stim.target_rec(-total_meas_per_round + i)
            # Use stored spatial coordinates; Relative Time is 0
            det_coord = [x_anc_coords[i][0], x_anc_coords[i][1], 0]
            
            if not is_first_round:
                prev_rec = stim.target_rec(-total_meas_per_round * 2 + i)
                c.append("DETECTOR", [current_rec, prev_rec], det_coord)
            elif memory_basis == "X":
                 c.append("DETECTOR", [current_rec], det_coord)

        # Z-Check Detectors
        for i in range(num_z_checks):
            offset = num_x_checks + i
            current_rec = stim.target_rec(-total_meas_per_round + offset)
            det_coord = [z_anc_coords[i][0], z_anc_coords[i][1], 0]

            if not is_first_round:
                prev_rec = stim.target_rec(-total_meas_per_round * 2 + offset)
                c.append("DETECTOR", [current_rec, prev_rec], det_coord)
            elif memory_basis == "Z":
                c.append("DETECTOR", [current_rec], det_coord)

    # --- 4. Build Schedule ---
    
    # First Round (Unrolled)
    # Detectors will be at t=0.0
    append_measurement_round(circuit, is_first_round=True)
    
    # Loop Body
    loop_body = stim.Circuit()
    append_measurement_round(loop_body, is_first_round=False)
    
    if rounds > 1:
        circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop_body))

    # --- 5. Final Readout ---
    # Detectors here will use the current time shift (e.g., t=1.0 for 1 round)
    if memory_basis == "Z":
        circuit.append("M", data_qubits) 
        for i, row in enumerate(Hz):
            rec_targets = []
            for data_idx in np.flatnonzero(row):
                rec_targets.append(stim.target_rec(-(num_data - data_idx)))
            rec_targets.append(stim.target_rec(-(num_data + num_z_checks - i)))
            
            det_coord = [z_anc_coords[i][0], z_anc_coords[i][1], 0]
            circuit.append("DETECTOR", rec_targets, det_coord)
            
    elif memory_basis == "X":
        circuit.append("MX", data_qubits)
        for i, row in enumerate(Hx):
            rec_targets = []
            for data_idx in np.flatnonzero(row):
                rec_targets.append(stim.target_rec(-(num_data - data_idx)))
            rec_targets.append(stim.target_rec(-(num_data + num_z_checks + num_x_checks - i)))
            
            det_coord = [x_anc_coords[i][0], x_anc_coords[i][1], 0]
            circuit.append("DETECTOR", rec_targets, det_coord)

    # --- 6. Logical Observable ---
    L_X, L_Z = get_logical_operators(Hx, Hz)
    logical_op = L_Z if memory_basis == "Z" else L_X
    
    obs_targets = []
    for k in np.flatnonzero(logical_op):
        obs_targets.append(stim.target_rec(-(num_data - k)))
    
    circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)

    return circuit

def generate_experiment_with_noise(
    Hx: np.ndarray, 
    Hz: np.ndarray, 
    rounds: int, 
    noise_model_name: str,
    noise_params: dict,
    memory_basis: str = "Z"
) -> stim.Circuit:
    # 1. Generate clean circuit
    clean_circuit = generate_css_memory_experiment(Hx, Hz, rounds, memory_basis=memory_basis)
    
    # 2. Identify data qubits
    num_data = Hx.shape[1]
    data_qubits = list(range(num_data))

    # 3. Apply Noise Model
    base_p = noise_params['p']

    if noise_model_name == "depolarizing":
        return standard_depolarizing_noise_model(
            circuit=clean_circuit,
            data_qubits=data_qubits,
            after_clifford_depolarization=noise_params.get('p_clifford', base_p),
            after_reset_flip_probability=noise_params.get('p_reset', base_p),
            before_measure_flip_probability=noise_params.get('p_meas', base_p),
            before_round_data_depolarization=noise_params.get('p_data_round', base_p)
        )
        
    elif noise_model_name == "si1000":
        return si1000_noise_model(
            circuit=clean_circuit,
            data_qubits=data_qubits,
            probability=base_p
        )
        
    elif noise_model_name == "bravyi":
        return bravyi_noise_model(
            circuit=clean_circuit,
            error_rate=base_p
        )
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