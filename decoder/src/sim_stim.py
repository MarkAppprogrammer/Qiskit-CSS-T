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

# --- Setup Paths ---
sys.path.append(".")
if not os.path.exists("convert_alist.py"):
    # Fallback for dev environment structure
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
        "default_n_list": [8, 18, 24],
        "alist_suffix": ".alist"
    },
    "dual_containing": {
        "dir": OP2_DIR,
        "dist_dict": OP2_DICT,
        "if_self_dual": False,
        "default_n_list": [7, 17, 23],
        "alist_suffix": ".alist"
    }
}

# --- Helper Functions ---

def get_parity_matrices(n: int, config: dict):
    """
    Reads alist files and returns Hx, Hz parity check matrices.
    """
    d = config['dist_dict'][n]
    base_dir = config['dir']
    if_self_dual = config['if_self_dual']
    suffix = config.get('alist_suffix', '.alist')
    
    F2 = galois.GF(2)
    
    # Special handling for n=17 (QR code specific split)
    if n == 17 and not if_self_dual:
        alistFilePathHz = os.path.join(base_dir, f"n{n}_d{d}_Hz.alist")
        Hz = F2(readAlist(alistFilePathHz))
        alistFilePathHx = os.path.join(base_dir, f"n{n}_d{d}_Hx.alist")
        Hx = F2(readAlist(alistFilePathHx))
    else:
        # Standard loading
        filename = f"n{n}_d{d}{suffix}"
        alistFilePath = os.path.join(base_dir, filename)
        
        if not os.path.exists(alistFilePath):
             # Try without suffix if failed, or adjust naming convention here
             raise FileNotFoundError(f"Could not find alist file: {alistFilePath}")

        GenMat = F2(readAlist(alistFilePath))
        
        if if_self_dual:
            # Puncture the last column for self-dual logic
            G_punctured = GenMat[:, :-1]  
            # For punctured code: H = NullSpace(G_punctured)
            Hz = Hx = G_punctured.null_space() 
        else:
            # Dual containing: H = NullSpace(G)
            Hz = Hx = GenMat.null_space()

    Hx = np.array(Hx, dtype=np.uint8)
    Hz = np.array(Hz, dtype=np.uint8)

    return Hx, Hz

def get_logical_operators(Hx, Hz):
    """
    Finds one pair of Logical X and Logical Z operators for a CSS code.
    Assumes one logical qubit for simplicity (k=1).
    """
    GF = galois.GF(2)
    gf_Hx = GF(Hx.astype(int))
    gf_Hz = GF(Hz.astype(int))

    lz_candidates = gf_Hx.null_space()
    lx_candidates = gf_Hz.null_space()

    # Heuristic: Pick the first basis vector of the null space
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
    memory_basis: str = "Z"
) -> stim.Circuit:
    
    num_data = Hx.shape[1]
    num_x_checks = Hx.shape[0]
    num_z_checks = Hz.shape[0]
    
    data_qubits = list(range(num_data))
    x_ancillas = list(range(num_data, num_data + num_x_checks))
    z_ancillas = list(range(num_data + num_x_checks, num_data + num_x_checks + num_z_checks))
    
    circuit = stim.Circuit()

    # --- Layout ---
    v_spacing = 2.0 
    data_h_spacing = 2.0
    ancilla_h_spacing = 4.0 
    data_width = (num_data - 1) * data_h_spacing
    data_center_x = data_width / 2.0
    x_anc_coords = []
    z_anc_coords = []

    # X-Ancillas (Top)
    x_start_x = get_start_x(num_x_checks, ancilla_h_spacing, data_center_x)
    for i, q in enumerate(x_ancillas):
        pos = [x_start_x + i * ancilla_h_spacing, 0]
        circuit.append("QUBIT_COORDS", [q], pos)
        x_anc_coords.append(pos)

    # Data Qubits (Middle)
    for i, q in enumerate(data_qubits):
        circuit.append("QUBIT_COORDS", [q], [i * data_h_spacing, v_spacing])

    # Z-Ancillas (Bottom)
    z_start_x = get_start_x(num_z_checks, ancilla_h_spacing, data_center_x)
    for i, q in enumerate(z_ancillas):
        pos = [z_start_x + i * ancilla_h_spacing, v_spacing * 2]
        circuit.append("QUBIT_COORDS", [q], pos)
        z_anc_coords.append(pos)

    # --- Initialization ---
    if memory_basis == "Z":
        circuit.append("R", data_qubits) 
    elif memory_basis == "X":
        circuit.append("RX", data_qubits) 
    
    circuit.append("R", x_ancillas + z_ancillas)
    circuit.append("TICK")

    # --- Round Function ---
    def append_measurement_round(c: stim.Circuit, is_first_round: bool):
        # X-Stabilizers
        c.append("H", x_ancillas)
        c.append("TICK")
        append_layered_cnots(c, Hx, x_ancillas, data_qubits, ancilla_is_control=True)
        c.append("H", x_ancillas)
        c.append("M", x_ancillas)
        c.append("TICK")
        
        # Z-Stabilizers
        append_layered_cnots(c, Hz, z_ancillas, data_qubits, ancilla_is_control=False)
        c.append("M", z_ancillas)
        c.append("R", x_ancillas + z_ancillas)

        # Detectors
        total_meas = num_x_checks + num_z_checks
        
        # X-Check Detectors
        for i in range(num_x_checks):
            current_rec = stim.target_rec(-total_meas + i)
            det_coord = [x_anc_coords[i][0], x_anc_coords[i][1], 0]
            if not is_first_round:
                prev_rec = stim.target_rec(-total_meas * 2 + i)
                c.append("DETECTOR", [current_rec, prev_rec], det_coord)
            elif memory_basis == "X":
                 c.append("DETECTOR", [current_rec], det_coord)

        # Z-Check Detectors
        for i in range(num_z_checks):
            offset = num_x_checks + i
            current_rec = stim.target_rec(-total_meas + offset)
            det_coord = [z_anc_coords[i][0], z_anc_coords[i][1], 0]
            if not is_first_round:
                prev_rec = stim.target_rec(-total_meas * 2 + offset)
                c.append("DETECTOR", [current_rec, prev_rec], det_coord)
            elif memory_basis == "Z":
                c.append("DETECTOR", [current_rec], det_coord)

    # --- Build Schedule ---
    append_measurement_round(circuit, is_first_round=True)
    loop_body = stim.Circuit()
    append_measurement_round(loop_body, is_first_round=False)
    if rounds > 1:
        circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop_body))

    # --- Readout ---
    if memory_basis == "Z":
        circuit.append("M", data_qubits) 
        for i, row in enumerate(Hz):
            rec_targets = []
            for data_idx in np.flatnonzero(row):
                rec_targets.append(stim.target_rec(-(num_data - data_idx)))
            rec_targets.append(stim.target_rec(-(num_data + num_z_checks - i)))
            circuit.append("DETECTOR", rec_targets, [z_anc_coords[i][0], z_anc_coords[i][1], 0])
            
    elif memory_basis == "X":
        circuit.append("MX", data_qubits)
        for i, row in enumerate(Hx):
            rec_targets = []
            for data_idx in np.flatnonzero(row):
                rec_targets.append(stim.target_rec(-(num_data - data_idx)))
            rec_targets.append(stim.target_rec(-(num_data + num_z_checks + num_x_checks - i)))
            circuit.append("DETECTOR", rec_targets, [x_anc_coords[i][0], x_anc_coords[i][1], 0])

    # --- Logical Observable ---
    L_X, L_Z = get_logical_operators(Hx, Hz)
    logical_op = L_Z if memory_basis == "Z" else L_X
    obs_targets = [stim.target_rec(-(num_data - k)) for k in np.flatnonzero(logical_op)]
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
    clean_circuit = generate_css_memory_experiment(Hx, Hz, rounds, memory_basis=memory_basis)
    data_qubits = list(range(Hx.shape[1]))
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
        return si1000_noise_model(circuit=clean_circuit, data_qubits=data_qubits, probability=base_p)
    elif noise_model_name == "bravyi":
        return bravyi_noise_model(circuit=clean_circuit, error_rate=base_p)
    else:
        raise ValueError(f"Unknown noise model: {noise_model_name}")

# --- Data Processing ---

def parse_and_average_stats(stats: List[sinter.TaskStats], trace_file: str, model_name: str) -> pd.DataFrame:
    """Parses Sinter stats and combines with MWPF decoding trace data."""
    avg_obj = 0.0
    avg_cpu = 0.0
    try:
        df_details = SinterMWPFDecoder.parse_mwpf_trace(trace_file)
        if not df_details.empty:
            avg_obj = df_details['objective_value'].mean()
            avg_cpu = df_details['cpu_time'].mean()
    except Exception as e:
        print(f"Warning: Could not parse trace file. {e}")
    
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

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--max_shots", type=int, default=100_000)
    parser.add_argument("--output", type=str, default="results.csv")
    parser.add_argument("--code_type", type=str, choices=["self_dual", "dual_containing"], required=True,
                        help="Choose 'self_dual' (Option 1) or 'dual_containing' (Option 2)")
    
    args = parser.parse_args()

    # Get Configuration
    selected_config = CODE_CONFIGS[args.code_type]
    n_list = selected_config["default_n_list"]
    
    # Simulation Parameters
    noise_values = np.logspace(-3, -1, 9)
    target_models = ["depolarizing"]
    
    output_base, output_ext = os.path.splitext(args.output)

    for model_name in target_models:
        print(f"\n{'='*60}")
        print(f"Running simulation for model: {model_name} | Code Type: {args.code_type}")
        print(f"{'='*60}")

        tasks = []
        for n in n_list:
            try:
                # Use the new helper function with config passed in
                Hx, Hz = get_parity_matrices(n, selected_config)
                d = selected_config['dist_dict'][n]
                num_rounds = d * 3 

                print(f"Generating tasks for n={n}, d={d}...")

                for p in noise_values:
                    circuit = generate_experiment_with_noise(
                        Hx=Hx, Hz=Hz, rounds=num_rounds, 
                        noise_model_name=model_name, 
                        noise_params={"p": p}
                    )
                    
                    tasks.append(
                        sinter.Task(
                            circuit=circuit,
                            decoder='mwpf',
                            json_metadata={
                                'n': n, 'd': d, 'r': num_rounds, 'p': p, 
                                'noise_model': model_name,
                                'code_type': args.code_type
                            }
                        )
                    )
            except Exception as e:
                print(f"Skipping n={n}: {e}")
                import traceback
                traceback.print_exc()

        if not tasks:
            print(f"No tasks generated for {model_name}. Skipping.")
            continue

        # Use a fresh temporary file for each model's trace
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            trace_path = tmp.name
        
        try:
            print(f"Initializing Decoder with trace file: {trace_path}")
            # Ensure cluster_node_limit matches your local installation's limits
            mwpf_decoder = SinterMWPFDecoder(cluster_node_limit=15, trace_filename=trace_path)

            print(f"Starting collection on {args.workers} workers...")
            collected_stats = sinter.collect(
                num_workers=args.workers,
                tasks=tasks,
                custom_decoders={"mwpf": mwpf_decoder},
                max_shots=args.max_shots,
                print_progress=True,
            )

            # Pass model_name here to fix the argument mismatch
            final_df = parse_and_average_stats(collected_stats, trace_path, model_name)
            
            output_filename = f"{output_base}_{model_name}{output_ext}"
            final_df.to_csv(output_filename, index=False)
            print(f"Results for {model_name} saved to {output_filename}")

        finally:
            if os.path.exists(trace_path):
                os.remove(trace_path)

if __name__ == "__main__":
    main()