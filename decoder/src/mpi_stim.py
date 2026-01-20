import sys
import os
import glob
import tempfile
import argparse
import numpy as np
import pandas as pd
import galois
import stim
import sinter
import traceback
from typing import List

# --- SETUP PATHS ---
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

# --- CONFIGURATION ---
BASE_ALIST_DIR_DEV = "../../doubling-CSST/alistMats/"
BASE_ALIST_DIR_LOCAL = "./alistMats/"
BASE_ALIST_PATH = BASE_ALIST_DIR_LOCAL if os.path.exists(BASE_ALIST_DIR_LOCAL) else BASE_ALIST_DIR_DEV

OP1_DIR = os.path.join(BASE_ALIST_PATH, "GO03_self_dual/")
OP1_DICT = {4:2, 6:2, 8:2, 10:2, 12:4, 14:4, 16:4, 18:4, 20:4, 22:6, 24:6, 26:6, 28:6, 30:6, 32:8, 34:6, 36:8, 38:8, 40:8, 42:8, 44:8, 46:8, 48:8, 50:8, 52:10, 54:8, 56:10, 58:10, 60:12, 62:10, 64:10}
OP2_DIR = os.path.join(BASE_ALIST_PATH, "QR_dual_containing/")
OP2_DICT = {7:3, 17:5, 23:7, 47:11, 79:15, 103:19, 167:23}

CODE_CONFIGS = {
    "self_dual": {
        "dir": OP1_DIR, "dist_dict": OP1_DICT, "if_self_dual": True, "default_n_list": [8, 18, 24, 48], "alist_suffix": ".alist"
    },
    "dual_containing": {
        "dir": OP2_DIR, "dist_dict": OP2_DICT, "if_self_dual": False, "default_n_list": [7, 17, 23, 47], "alist_suffix": ".alist"
    }
}

# --- HELPER FUNCTIONS ---

def get_parity_matrices(n: int, config: dict):
    d = config['dist_dict'][n]
    base_dir = config['dir']
    if_self_dual = config['if_self_dual']
    suffix = config.get('alist_suffix', '.alist')
    F2 = galois.GF(2)
    
    if n == 17 and not if_self_dual:
        Hz = F2(readAlist(os.path.join(base_dir, f"n{n}_d{d}_Hz.alist")))
        Hx = F2(readAlist(os.path.join(base_dir, f"n{n}_d{d}_Hx.alist")))
    else:
        filename = f"n{n}_d{d}{suffix}"
        alistFilePath = os.path.join(base_dir, filename)
        if not os.path.exists(alistFilePath): raise FileNotFoundError(f"Missing: {alistFilePath}")
        GenMat = F2(readAlist(alistFilePath))
        if if_self_dual:
            G_punctured = GenMat[:, :-1]; Hz = Hx = G_punctured.null_space() 
        else:
            Hz = Hx = GenMat.null_space()
    return np.array(Hx, dtype=np.uint8), np.array(Hz, dtype=np.uint8)

def get_start_x(num_items, spacing, center):
    return center - ((num_items - 1) * spacing / 2.0) if num_items > 1 else center

def append_layered_cnots(c, check_matrix, ancilla_idxs, data_idxs, ancilla_is_control):
    check_targets = []; max_weight = 0
    for row in check_matrix:
        targets = [data_idxs[q] for q in np.flatnonzero(row)]
        check_targets.append(targets); max_weight = max(max_weight, len(targets))
    for k in range(max_weight):
        layer_args = []
        for i, targets in enumerate(check_targets):
            if k < len(targets):
                anc = ancilla_idxs[i]; dat = targets[k]
                layer_args.extend([anc, dat] if ancilla_is_control else [dat, anc])
        if layer_args: c.append("CX", layer_args); c.append("TICK")

def generate_css_memory_experiment(Hx, Hz, rounds, memory_basis="Z"):
    num_data = Hx.shape[1]; num_x_checks = Hx.shape[0]; num_z_checks = Hz.shape[0]
    data_qubits = list(range(num_data))
    x_ancillas = list(range(num_data, num_data + num_x_checks))
    z_ancillas = list(range(num_data + num_x_checks, num_data + num_x_checks + num_z_checks))
    
    circuit = stim.Circuit()
    x_anc_coords = []; z_anc_coords = []
    
    # Layout
    x_start = get_start_x(num_x_checks, 4.0, ((num_data - 1) * 2.0)/2.0)
    for i, q in enumerate(x_ancillas):
        pos = [x_start + i * 4.0, 0]; circuit.append("QUBIT_COORDS", [q], pos); x_anc_coords.append(pos)
    for i, q in enumerate(data_qubits):
        circuit.append("QUBIT_COORDS", [q], [i * 2.0, 2.0])
    z_start = get_start_x(num_z_checks, 4.0, ((num_data - 1) * 2.0)/2.0)
    for i, q in enumerate(z_ancillas):
        pos = [z_start + i * 4.0, 4.0]; circuit.append("QUBIT_COORDS", [q], pos); z_anc_coords.append(pos)

    if memory_basis == "Z": circuit.append("R", data_qubits) 
    elif memory_basis == "X": circuit.append("RX", data_qubits) 
    circuit.append("R", x_ancillas + z_ancillas); circuit.append("TICK")

    def append_round(c, first):
        c.append("H", x_ancillas); c.append("TICK")
        append_layered_cnots(c, Hx, x_ancillas, data_qubits, True)
        c.append("H", x_ancillas); c.append("M", x_ancillas); c.append("TICK")
        append_layered_cnots(c, Hz, z_ancillas, data_qubits, False)
        c.append("M", z_ancillas); c.append("R", x_ancillas + z_ancillas)
        tot = num_x_checks + num_z_checks
        for i in range(num_x_checks):
            t = stim.target_rec; args = [t(-tot+i), t(-tot*2+i)] if not first else [t(-tot+i)]
            if not first or memory_basis == "X": c.append("DETECTOR", args, [x_anc_coords[i][0], x_anc_coords[i][1], 0])
        for i in range(num_z_checks):
            offset = num_x_checks + i; t = stim.target_rec; args = [t(-tot+offset), t(-tot*2+offset)] if not first else [t(-tot+offset)]
            if not first or memory_basis == "Z": c.append("DETECTOR", args, [z_anc_coords[i][0], z_anc_coords[i][1], 0])

    append_round(circuit, True)
    if rounds > 1:
        loop = stim.Circuit(); append_round(loop, False)
        circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop))

    if memory_basis == "Z":
        circuit.append("M", data_qubits)
        for i, row in enumerate(Hz):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + num_z_checks - i)))
            circuit.append("DETECTOR", rec, [z_anc_coords[i][0], z_anc_coords[i][1], 0])
    elif memory_basis == "X":
        circuit.append("MX", data_qubits)
        for i, row in enumerate(Hx):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + num_z_checks + num_x_checks - i)))
            circuit.append("DETECTOR", rec, [x_anc_coords[i][0], x_anc_coords[i][1], 0])

    GF = galois.GF(2); lx = GF(Hz.astype(int)).null_space()[0]; lz = GF(Hx.astype(int)).null_space()[0]
    op = lz if memory_basis == "Z" else lx
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(num_data - k)) for k in np.flatnonzero(op)], 0)
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


def parse_and_average_stats(stats: List[sinter.TaskStats], trace_file: str, model_name: str) -> pd.DataFrame:
    try:
        # Calls the static method on the imported class
        df_details = SinterMWPFDecoder.parse_mwpf_trace(trace_file)
        if not df_details.empty:
            avg_obj = df_details['objective_value'].mean()
            avg_cpu = df_details['cpu_time'].mean()
        else: avg_obj = 0.0; avg_cpu = 0.0
    except Exception: avg_obj = 0.0; avg_cpu = 0.0
    
    results = []
    for s in stats:
        m = s.json_metadata
        results.append({
            'noise_model': model_name, 'n': m['n'], 'd': m['d'], 'r': m['r'], 'p': m['p'],
            'code_type': m.get('code_type', 'unknown'), 'shots': s.shots, 'errors': s.errors,
            'total_logical_error_rate': s.errors / s.shots if s.shots > 0 else 0,
            'mean_objective_value': avg_obj, 'average_cpu_time_seconds': avg_cpu,
        })
    return pd.DataFrame(results)

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    default_workers = int(slurm_cpus) if slurm_cpus else os.cpu_count()
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--max_shots", type=int, default=1_000_000)
    parser.add_argument("--output", type=str, default="hyperion_results.csv")
    parser.add_argument("--code_type", type=str, choices=["self_dual", "dual_containing"], required=True)
    args = parser.parse_args()

    proc_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    selected_config = CODE_CONFIGS[args.code_type]
    n_list = selected_config["default_n_list"]
    noise_values = np.logspace(-4, -2, 10).tolist()
    models_to_run = ["depolarizing", "si1000"]
    output_base, output_ext = os.path.splitext(args.output)

    for model_name in models_to_run:
        all_tasks = []
        for n in n_list:
            try:
                Hx, Hz = get_parity_matrices(n, selected_config)
                d = selected_config['dist_dict'][n]
                for p in noise_values:
                    circuit = generate_experiment_with_noise(Hx, Hz, d*3, model_name, {"p": p})
                    all_tasks.append(sinter.Task(circuit=circuit, decoder='mwpf', 
                        json_metadata={'n': n, 'd': d, 'r': d*3, 'p': p, 'noise_model': model_name, 'code_type': args.code_type}))
            except Exception as e: print(f"[Node {proc_rank}] Skipping n={n}: {e}")

        # Distribute tasks
        my_tasks = [t for i, t in enumerate(all_tasks) if i % world_size == proc_rank]
        if not my_tasks: continue
        
        # Calculate print interval to avoid flooding logs (e.g., update 10 times total)
        total_tasks = len(my_tasks)
        print_interval = max(1, total_tasks // 10)
        
        print(f"[Node {proc_rank}] Processing {total_tasks} tasks for {model_name}...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            trace_path = tmp.name
            
        try:
            mwpf_decoder = SinterMWPFDecoder(cluster_node_limit=50, trace_filename=trace_path)
            
            collected_stats = []
            
            iterator = sinter.iter_collect(
                num_workers=args.workers, 
                tasks=my_tasks, 
                custom_decoders={"mwpf": mwpf_decoder},
                max_shots=args.max_shots
            )
            
            # Manual loop to gather stats and print sparsely
            for i, stat in enumerate(iterator):
                collected_stats.append(stat)
                
                # Print only at specific intervals or at the very end
                if (i + 1) % print_interval == 0 or (i + 1) == total_tasks:
                    print(f"[Node {proc_rank}] {model_name} Progress: {i + 1}/{total_tasks} completed.", flush=True)
            # ----------------------------------------

            final_df = parse_and_average_stats(collected_stats, trace_path, model_name)
            
            part_filename = f"{output_base}_{model_name}_rank{proc_rank}{output_ext}"
            final_df.to_csv(part_filename, index=False)
            print(f"[Node {proc_rank}] Saved {part_filename}")

        finally:
            if os.path.exists(trace_path):
                for f in glob.glob(f"{trace_path}*"):
                    try: os.remove(f)
                    except OSError: pass

if __name__ == "__main__":
    main()