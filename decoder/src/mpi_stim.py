import sys
import os
import argparse
import numpy as np
import pandas as pd
import galois
import stim
import sinter
import traceback
import json
import time
from typing import List, Tuple, Dict, Optional

# --- SETUP PATHS ---
sys.path.append(".")
if not os.path.exists("convert_alist.py"):
    sys.path.append('../../../doubling-CSST/')

try:
    from convert_alist import readAlist
    from mwpf.sinter_decoders import SinterMWPFDecoder 
    from noise_models import (
        standard_depolarizing_noise_model, 
        si1000_noise_model
    )
except ImportError:
    pass

# --- CONFIGURATION ---
BASE_ALIST_DIR_DEV = "../../../doubling-CSST/alistMats/"
BASE_ALIST_DIR_LOCAL = "./alistMats/"
BASE_ALIST_DIR_PROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "doubling-CSST/alistMats/")
BASE_ALIST_PATH = BASE_ALIST_DIR_LOCAL if os.path.exists(BASE_ALIST_DIR_LOCAL) else (BASE_ALIST_DIR_PROJECT if os.path.exists(BASE_ALIST_DIR_PROJECT) else BASE_ALIST_DIR_DEV)

OP1_DIR = os.path.join(BASE_ALIST_PATH, "GO03_self_dual/")
OP1_DICT = {4:2, 6:2, 8:2, 10:2, 12:4, 14:4, 16:4, 18:4, 20:4, 22:6, 24:6, 26:6, 28:6, 30:6, 32:8, 34:6, 36:8, 38:8, 40:8, 42:8, 44:8, 46:8, 48:8, 50:8, 52:10, 54:8, 56:10, 58:10, 60:12, 62:10, 64:10}

OP2_DIR = os.path.join(BASE_ALIST_PATH, "QR_dual_containing/")
OP2_DICT = {7:3, 17:5, 23:7, 47:11, 79:15, 103:19, 167:23}

CODE_CONFIGS = {
    "self_dual": {
        "dir": OP1_DIR, "dist_dict": OP1_DICT, "if_self_dual": True, "n_list": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64], "alist_suffix": ".alist"
    },
    "dual_containing": {
        "dir": OP2_DIR, "dist_dict": OP2_DICT, "if_self_dual": True, "n_list": [7, 23, 47, 79, 103, 167], "alist_suffix": ".alist"
    }
}

# --- SCHEDULE LOADING ---
def load_schedule(filepath):
    """Loads a CNOT schedule from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    schedule = {}
    for k, v in data.items():
        parts = k.split(',')
        schedule[(int(parts[0]), int(parts[1]))] = v
    return schedule

# --- CIRCUIT GENERATION ---

def get_start_x(num_items, spacing, center):
    return center - ((num_items - 1) * spacing / 2.0) if num_items > 1 else center

def append_scheduled_cnots(c: stim.Circuit, check_matrix: np.ndarray, 
                           ancilla_idxs: List[int], data_idxs: List[int], 
                           ancilla_is_control: bool, 
                           schedule: Optional[Dict[Tuple[int, int], int]]):
    ops = [] 
    for i, row in enumerate(check_matrix):
        targets = [data_idxs[q] for q in np.flatnonzero(row)]
        anc = ancilla_idxs[i]
        for t_idx, dat in enumerate(targets):
            key = (i, data_idxs.index(dat))
            priority = schedule.get(key, t_idx) if schedule else t_idx
            if ancilla_is_control: ops.append((priority, anc, dat))
            else: ops.append((priority, dat, anc))
            
    ops.sort(key=lambda x: x[0])
    
    current_prio = -100
    for prio, ctrl, targ in ops:
        c.append("CNOT", [ctrl, targ])
        current_prio = prio

def generate_css_memory_experiment(Hx, Hz, rounds, memory_basis="Z", cnot_schedule=None):
    is_self_dual = (Hx.shape == Hz.shape) and np.array_equal(Hx, Hz)
    if is_self_dual:
        return _generate_self_dual_schedule(Hx, rounds, memory_basis, cnot_schedule)
    else:
        return _generate_standard_schedule(Hx, Hz, rounds, memory_basis, cnot_schedule)

def _generate_self_dual_schedule(H, rounds, memory_basis, schedule):
    num_data = H.shape[1]; num_checks = H.shape[0]
    data_qubits = list(range(num_data)); ancillas = list(range(num_data, num_data + num_checks))
    circuit = stim.Circuit()
    
    for i, q in enumerate(data_qubits): circuit.append("QUBIT_COORDS", [q], [i * 2.0, 2.0])
    anc_start_x = get_start_x(num_checks, 4.0, ((num_data - 1) * 2.0) / 2.0)
    anc_coords = []
    for i, q in enumerate(ancillas):
        pos = [anc_start_x + i * 4.0, 4.0]
        circuit.append("QUBIT_COORDS", [q], pos); anc_coords.append(pos)

    circuit.append("R" if memory_basis == "Z" else "RX", data_qubits)
    circuit.append("R", ancillas); circuit.append("TICK")

    def append_round(c, is_first):
        c.append("H", ancillas); c.append("TICK")
        append_scheduled_cnots(c, H, ancillas, data_qubits, True, schedule)
        c.append("H", ancillas); c.append("M", ancillas); c.append("R", ancillas); c.append("TICK")
        append_scheduled_cnots(c, H, ancillas, data_qubits, False, schedule)
        c.append("M", ancillas); c.append("R", ancillas); c.append("TICK")
        
        total_m = 2 * num_checks
        for i in range(num_checks): 
            rec_now = stim.target_rec(-total_m + i); rec_prev = stim.target_rec(-total_m * 2 + i)
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "X": c.append("DETECTOR", args, [anc_coords[i][0], anc_coords[i][1], 0])
        for i in range(num_checks): 
            rec_now = stim.target_rec(-num_checks + i); rec_prev = stim.target_rec(-num_checks - total_m + i)
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "Z": c.append("DETECTOR", args, [anc_coords[i][0], anc_coords[i][1], 1])

    append_round(circuit, True)
    if rounds > 1:
        loop = stim.Circuit(); append_round(loop, False); circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop))

    if memory_basis == "Z":
        circuit.append("M", data_qubits)
        for i, row in enumerate(H):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + num_checks - i))) 
            circuit.append("DETECTOR", rec, [anc_coords[i][0], anc_coords[i][1], 1])
    elif memory_basis == "X":
        circuit.append("MX", data_qubits)
        for i, row in enumerate(H):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + 2*num_checks - i)))
            circuit.append("DETECTOR", rec, [anc_coords[i][0], anc_coords[i][1], 0])

    op = find_logical_operator(H, H, basis=memory_basis)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(num_data - k)) for k in np.flatnonzero(op)], 0)
    return circuit

def _generate_standard_schedule(Hx, Hz, rounds, memory_basis, schedule):
    num_data = Hx.shape[1]; num_x = Hx.shape[0]; num_z = Hz.shape[0]
    data_qubits = list(range(num_data))
    x_ancillas = list(range(num_data, num_data + num_x))
    z_ancillas = list(range(num_data + num_x, num_data + num_x + num_z))
    circuit = stim.Circuit()
    
    x_start = get_start_x(num_x, 4.0, ((num_data - 1) * 2.0)/2.0); x_coords = []
    for i, q in enumerate(x_ancillas): pos = [x_start + i * 4.0, 0]; circuit.append("QUBIT_COORDS", [q], pos); x_coords.append(pos)
    for i, q in enumerate(data_qubits): circuit.append("QUBIT_COORDS", [q], [i * 2.0, 2.0])
    z_start = get_start_x(num_z, 4.0, ((num_data - 1) * 2.0)/2.0); z_coords = []
    for i, q in enumerate(z_ancillas): pos = [z_start + i * 4.0, 4.0]; circuit.append("QUBIT_COORDS", [q], pos); z_coords.append(pos)

    circuit.append("R" if memory_basis == "Z" else "RX", data_qubits)
    circuit.append("R", x_ancillas + z_ancillas); circuit.append("TICK")

    def append_round(c, is_first):
        c.append("H", x_ancillas); c.append("TICK")
        append_scheduled_cnots(c, Hx, x_ancillas, data_qubits, True, schedule)
        c.append("H", x_ancillas); c.append("M", x_ancillas); c.append("TICK")
        append_scheduled_cnots(c, Hz, z_ancillas, data_qubits, False, schedule)
        c.append("M", z_ancillas); c.append("R", x_ancillas + z_ancillas)
        tot = num_x + num_z
        for i in range(num_x):
            rec = stim.target_rec(-tot + i); rec_prev = stim.target_rec(-tot * 2 + i)
            args = [rec, rec_prev] if not is_first else [rec]
            if not is_first or memory_basis == "X": c.append("DETECTOR", args, [x_coords[i][0], x_coords[i][1], 0])
        for i in range(num_z):
            offset = num_x + i
            rec = stim.target_rec(-tot + offset); rec_prev = stim.target_rec(-tot * 2 + offset)
            args = [rec, rec_prev] if not is_first else [rec]
            if not is_first or memory_basis == "Z": c.append("DETECTOR", args, [z_coords[i][0], z_coords[i][1], 0])

    append_round(circuit, True)
    if rounds > 1: loop = stim.Circuit(); append_round(loop, False); circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop))
    if memory_basis == "Z":
        circuit.append("M", data_qubits)
        for i, row in enumerate(Hz):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + num_z - i)))
            circuit.append("DETECTOR", rec, [z_coords[i][0], z_coords[i][1], 0])
    elif memory_basis == "X":
        circuit.append("MX", data_qubits)
        for i, row in enumerate(Hx):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + num_z + num_x - i)))
            circuit.append("DETECTOR", rec, [x_coords[i][0], x_coords[i][1], 0])

    op = find_logical_operator(Hx, Hz, basis=memory_basis)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(num_data - k)) for k in np.flatnonzero(op)], 0)
    return circuit

# --- HELPERS ---
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
        G_punctured = GenMat[:, :-1]
        Hz = Hx = G_punctured.null_space() 
    return np.array(Hx, dtype=np.uint8), np.array(Hz, dtype=np.uint8)

def find_logical_operator(Hx, Hz, basis="Z"):
    F2 = galois.GF(2)
    gf_Hx = F2(Hx); gf_Hz = F2(Hz)
    candidates = gf_Hx.null_space() if basis == "Z" else gf_Hz.null_space()
    stabilizers = gf_Hz if basis == "Z" else gf_Hx
    current_rank = np.linalg.matrix_rank(stabilizers)
    for cand in candidates:
        if np.linalg.matrix_rank(np.vstack([stabilizers, cand])) > current_rank:
            return np.array(cand, dtype=np.uint8)
    raise ValueError(f"Could not find a Logical {basis} operator!")

def generate_experiment_with_noise(Hx, Hz, rounds, noise_model_name, noise_params, memory_basis="Z", schedule=None):
    clean_circuit = generate_css_memory_experiment(Hx, Hz, rounds, memory_basis=memory_basis, cnot_schedule=schedule)
    num_data = Hx.shape[1]
    data_qubits = list(range(num_data))
    base_p = noise_params['p']
    if noise_model_name == "depolarizing":
        return standard_depolarizing_noise_model(
            circuit=clean_circuit, data_qubits=data_qubits,
            after_clifford_depolarization=noise_params.get('p_clifford', base_p),
            after_reset_flip_probability=noise_params.get('p_reset', base_p),
            before_measure_flip_probability=noise_params.get('p_meas', base_p),
            before_round_data_depolarization=noise_params.get('p_data_round', base_p)
        )
    elif noise_model_name == "si1000":
        return si1000_noise_model(circuit=clean_circuit, data_qubits=data_qubits, probability=base_p)
    else: raise ValueError(f"Unknown noise model: {noise_model_name}")

def parse_and_average_stats(stats: List[sinter.TaskStats], trace_file: str, model_name: str) -> pd.DataFrame:
    avg_obj = 0.0; avg_cpu = 0.0
    
    # --- FIX IS HERE: Remove 'os.path.exists(trace_file)' check ---
    # The workers create 'trace_file.PID', not 'trace_file'.
    # SinterMWPFDecoder.parse_mwpf_trace uses glob() to find the PID files.
    
    try:
        df_details = SinterMWPFDecoder.parse_mwpf_trace(trace_file)
        if not df_details.empty:
            avg_obj = df_details['objective_value'].mean()
            avg_cpu = df_details['cpu_time'].mean()
        # else:
            # print(f"DEBUG: Trace DF is empty for {trace_file}")
    except Exception as e:
        # print(f"DEBUG: Parse failed: {e}")
        pass

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
    is_slurm_mode = slurm_cpus is not None
    
    default_workers = int(slurm_cpus) if is_slurm_mode else os.cpu_count()
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--max_shots", type=int, default=1_000_000)
    parser.add_argument("--output", type=str, default="results.csv")
    parser.add_argument("--code_type", type=str, choices=["self_dual", "dual_containing"], required=True)
    args = parser.parse_args()

    proc_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    selected_config = CODE_CONFIGS[args.code_type]
    n_list = selected_config["n_list"]
    noise_values = np.logspace(-5, -3, 10).tolist()
    models_to_run = ["depolarizing"] 

    ver_dir = os.path.dirname(args.output)
    if not ver_dir: ver_dir = "."
    results_dir = os.path.join(ver_dir, "results")
    tmp_dir = os.path.join(ver_dir, "tmp")
    sched_dir = os.path.join(ver_dir, "schedules_cache")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(sched_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(args.output))[0]
    output_ext = os.path.splitext(args.output)[1]

    for model_name in models_to_run:
        all_tasks = []
        for n in n_list:
            try:
                Hx, Hz = get_parity_matrices(n, selected_config)
                d = selected_config['dist_dict'][n]
                
                # --- LOAD PRE-COMPUTED SCHEDULE ---
                schedule_file = os.path.join(sched_dir, f"sched_{args.code_type}_n{n}.json")
                best_schedule = None

                if os.path.exists(schedule_file):
                    if proc_rank == 0:
                        print(f"[Node 0] Loading pre-computed schedule for n={n}...")
                    best_schedule = load_schedule(schedule_file)
                else:
                    if proc_rank == 0:
                        print(f"[Node 0] WARNING: No schedule found for n={n}. Using default sequential schedule.")
                    
                    best_schedule = {}
                    for i, row in enumerate(Hz):
                        targets = [d for d in np.flatnonzero(row)]
                        for priority, data_idx in enumerate(targets):
                            best_schedule[(i, data_idx)] = priority

                for p in noise_values:
                    circuit = generate_experiment_with_noise(
                        Hx, Hz, d*5, model_name, {"p": p}, schedule=best_schedule
                    )
                    all_tasks.append(sinter.Task(
                        circuit=circuit, decoder='mwpf',
                        json_metadata={'n': n, 'd': d, 'r': d*5, 'p': p, 'noise_model': model_name, 'code_type': args.code_type}
                    ))
            except Exception as e: 
                print(f"[Node {proc_rank}] Skipping n={n}: {e}")
                traceback.print_exc()

        my_tasks = [t for i, t in enumerate(all_tasks) if i % world_size == proc_rank]
        if not my_tasks: continue
        
        total_expected_shots = len(my_tasks) * args.max_shots
        resume_path = os.path.join(tmp_dir, f"resume_{base_filename}_{model_name}_{proc_rank}.sinter")
        trace_path = os.path.join(tmp_dir, f"trace_{base_filename}_{model_name}_{proc_rank}.bin")
        part_filename = os.path.join(results_dir, f"{base_filename}_{model_name}_rank{proc_rank}{output_ext}")

        existing_data = []
        if os.path.exists(resume_path):
            try: existing_data = sinter.stats_from_csv_files(resume_path)
            except Exception: pass
        else:
            # We don't remove trace_path here because we want to allow restarts if possible,
            # but mainly because trace_path isn't a real file, it's a prefix.
            # Clean up old worker files if starting fresh:
            for old_f in __import__("glob").glob(f"{trace_path}*"):
                try: os.remove(old_f)
                except: pass

        try:
            mwpf_decoder = SinterMWPFDecoder(cluster_node_limit=50, trace_filename=trace_path)
            iterator = sinter.iter_collect(
                num_workers=args.workers, 
                tasks=my_tasks, 
                custom_decoders={"mwpf": mwpf_decoder},
                max_shots=args.max_shots,
                additional_existing_data=existing_data,
                max_batch_seconds=30,
                max_batch_size=1000
            )
            
            last_save_time = 0
            
            last_csv_update_time = time.time()
            csv_update_interval = 60
            
            recent_params = set()
            with open(resume_path, 'a') as resume_file:
                if resume_file.tell() == 0: print(sinter.CSV_HEADER, file=resume_file)
                for progress in iterator:
                    for stat in progress.new_stats:
                        print(stat.to_csv_line(), file=resume_file, flush=True)
                        recent_params.add(f"n={stat.json_metadata.get('n')}")
                    
                    if time.time() - last_csv_update_time > csv_update_interval:
                        try:
                            resume_file.flush()
                            os.fsync(resume_file.fileno())
                            current_stats = sinter.stats_from_csv_files(resume_path)
                            temp_df = parse_and_average_stats(current_stats, trace_path, model_name)
                            temp_df.to_csv(part_filename, index=False)
                            last_csv_update_time = time.time()
                        except Exception as e:
                            print(f"[Node {proc_rank}] ⚠️ Intermediate save failed: {e}")

                    if (time.time() - last_save_time > 10):
                        if not is_slurm_mode: print(progress.status_message, flush=True)
                        last_save_time = time.time()

            full_stats = sinter.stats_from_csv_files(resume_path)
            try:
                final_df = parse_and_average_stats(full_stats, trace_path, model_name)
                final_df.to_csv(part_filename, index=False)
                print(f"[Node {proc_rank}] ✅ Finished {model_name}.")
            except Exception as e: print(f"[Node {proc_rank}] ⚠️ Failed parsing: {e}")
        except Exception as e: traceback.print_exc()

if __name__ == "__main__":
    main()