import sys
import os
import argparse
import json
import numpy as np
import stim
import itertools
import random
from multiprocessing import Pool, cpu_count
import traceback

sys.path.append(".")
# Fallback import for convert_alist
if not os.path.exists("convert_alist.py"):
    sys.path.append('../../../doubling-CSST/')

try:
    from convert_alist import readAlist
except ImportError:
    pass

from helpers import get_parity_matrices, find_logical_operator, CODE_CONFIGS
from circuit import generate_css_memory_experiment
from noise_models import standard_depolarizing_noise_model

def get_effective_distance(circuit: stim.Circuit) -> int:
    """
    Attempts to calculate the effective code distance using Stim's graph analysis.
    NOTE: This often fails for 3D codes (Hypergraph codes) where errors trigger >2 detectors.
    """
    try:
        # decompose_errors=True attempts to break hyperedges, but isn't magic.
        # ignore_ungraphlike_errors=True allows the code to run even if some errors are complex,
        # but if ALL errors are complex (common in 3D), this might return a weird result.
        return len(circuit.shortest_graphlike_error(
            ignore_ungraphlike_errors=True,
            decompose_errors=True
        ))
    except ValueError:
        # This usually implies the error graph is empty or disconnected due to hyperedges
        return 0

def evaluate_schedule(args):
    Hx, Hz, rounds, perm_indices, adjacency, data_qubits = args
    
    schedule = {}
    for anc_idx, targets in adjacency.items():
        current_perm = perm_indices[:len(targets)]
        for priority, target_index_in_list in enumerate(current_perm):
            if target_index_in_list < len(targets):
                data_idx = targets[target_index_in_list]
                schedule[(anc_idx, data_idx)] = priority

    try:
        # Generate circuit
        circuit = generate_css_memory_experiment(
            Hx, Hz, rounds=rounds, memory_basis="Z", schedule=schedule
        )
        
        # Add noise to create the error graph
        noisy_circuit = standard_depolarizing_noise_model(
            circuit=circuit,
            data_qubits=data_qubits,
            after_clifford_depolarization=0.001,
            after_reset_flip_probability=0.001,
            before_measure_flip_probability=0.001,
            before_round_data_depolarization=0.001
        )
        
        dist = get_effective_distance(noisy_circuit)
        return dist, schedule
    except Exception:
        # If construction fails entirely, return 0
        return 0, schedule

def optimize_code_schedule(val, config, max_attempts=500):
    print(f"\n--- Optimizing val={val} ({config.get('type', 'local')}) ---")
    
    # 1. Load Matrices
    try:
        Hx, Hz = get_parity_matrices(val, config)
    except Exception as e:
        print(f"Failed to load matrices for val={val}: {e}")
        return {}, 0

    num_data = Hx.shape[1]
    data_qubits = list(range(num_data))
    
    # 2. Determine Parameters
    if config.get('source') == 'qcodeplot3d':
        d = val
    else:
        d = config['dist_dict'].get(val, 3)

    # 3. Analyze Stabilizers
    adjacency = {}
    max_weight = 0
    weights = []
    for i, row in enumerate(Hz):
        targets = [data_qubits[d_idx] for d_idx in np.flatnonzero(row)]
        adjacency[i] = targets
        w = len(targets)
        max_weight = max(max_weight, w)
        weights.append(w)
    
    print(f"  Stabilizer weights: avg={np.mean(weights):.1f}, max={max_weight}")
    
    # 4. Generate Candidates
    candidates = []
    # Deterministic: Sequential
    candidates.append(tuple(range(max_weight)))
    # Deterministic: Reverse
    candidates.append(tuple(range(max_weight-1, -1, -1)))
    # Deterministic: Interleaved (Good for lattice interference)
    if max_weight >= 4:
        pairs = []
        for i in range(0, max_weight - 1, 2):
            pairs.extend([i, i+1] if i % 4 == 0 else [i+1, i])
        if max_weight % 2 == 1: pairs.append(max_weight - 1)
        candidates.append(tuple(pairs))

    # Random fills
    remaining_slots = max_attempts - len(candidates)
    if max_weight <= 7:
        all_perms = list(itertools.permutations(range(max_weight)))
        all_perms = [p for p in all_perms if p not in candidates]
        if len(all_perms) > remaining_slots:
            candidates.extend(random.sample(all_perms, remaining_slots))
        else:
            candidates.extend(all_perms)
    else:
        for _ in range(remaining_slots):
            p = list(range(max_weight))
            random.shuffle(p)
            candidates.append(tuple(p))

    # 5. Evaluate Candidates
    rounds = d + 1 # Sufficient to detect errors
    worker_args = [(Hx, Hz, rounds, p, adjacency, data_qubits) for p in candidates]
    
    best_dist = -1
    best_schedule = None
    
    # Run parallel evaluation
    num_procs = min(cpu_count(), 20)
    with Pool(processes=num_procs) as pool:
        results = pool.map(evaluate_schedule, worker_args)
        
    for dist, sched in results:
        if dist > best_dist:
            best_dist = dist
            best_schedule = sched
    
    # 6. Fallback Logic for 3D/Hypergraph Codes
    is_3d = config.get('source') == 'qcodeplot3d'
    
    if best_dist == 0:
        print(f"  Warning: Effective distance is 0. This is expected for 3D codes due to hypergraph errors.")
        print(f"  -> Defaulting to 'Sequential' schedule to ensure simulation can proceed.")
        
        # Find the sequential schedule (it was the first candidate added)
        # We re-construct it to be sure
        seq_perm = tuple(range(max_weight))
        # Re-map the sequential perm to the schedule format
        best_schedule = {}
        for anc_idx, targets in adjacency.items():
            current_perm = seq_perm[:len(targets)]
            for priority, target_index_in_list in enumerate(current_perm):
                if target_index_in_list < len(targets):
                    data_idx = targets[target_index_in_list]
                    best_schedule[(anc_idx, data_idx)] = priority
        
        best_dist = 0 # It remains 0/Unknown

    print(f"  Selected Schedule Distance: {best_dist} (Theory={d})")
    return best_schedule, best_dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_type", type=str, choices=CODE_CONFIGS.keys(), required=True)
    parser.add_argument("--vals", type=int, nargs='+', help="List of values (n or d) to optimize")
    parser.add_argument("--output_dir", type=str, default="schedules_cache")
    parser.add_argument("--max_attempts", type=int, default=200)
    args = parser.parse_args()

    config = CODE_CONFIGS[args.code_type]
    target_vals = args.vals if args.vals else config.get('iter_list', config.get('n_list'))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    summary = []

    for val in target_vals:
        if config.get('source', 'local') == 'local' and val not in config['dist_dict']:
            print(f"Skipping val={val} (No config info)")
            continue
            
        schedule, eff_dist = optimize_code_schedule(val, config, args.max_attempts)
        
        if schedule:
            filename = f"sched_{args.code_type}_val{val}.json"
            filepath = os.path.join(args.output_dir, filename)
            
            serializable = {f"{k[0]},{k[1]}": v for k, v in schedule.items()}
            with open(filepath, 'w') as f:
                json.dump(serializable, f)
            print(f"  Saved to {filepath}")
            
            if config.get('source') == 'qcodeplot3d':
                theo_d = val
            else:
                theo_d = config['dist_dict'].get(val, '?')
                
            summary.append((val, eff_dist, theo_d))
        else:
            print(f"  Failed to generate schedule for val={val}")

    print("\n--- Summary ---")
    print(f"{'Val':<6} | {'Eff. D':<8} | {'Theor. D':<8}")
    print("-" * 28)
    for v, ed, td in summary:
        print(f"{v:<6} | {ed:<8} | {td:<8}")

if __name__ == "__main__":
    main()