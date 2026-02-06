import sys
import os
import argparse
import json
import numpy as np
import stim
import itertools
import random
import math
from multiprocessing import Pool, cpu_count

# --- IMPORT SETUP ---
sys.path.append(".")
if not os.path.exists("convert_alist.py"):
    sys.path.append('../../../doubling-CSST/')

try:
    from convert_alist import readAlist
    from mpi_stim import (
        get_parity_matrices,
        CODE_CONFIGS,
        generate_css_memory_experiment
    )
    from noise_models import standard_depolarizing_noise_model
except ImportError:
    # Fallback if mpi_stim unavailable, though get_parity_matrices is needed
    pass

# --- CORE METRIC ---

def get_effective_distance(circuit: stim.Circuit) -> int:
    try:
        # decompose_errors=True is critical for accuracy
        # ignore_ungraphlike_errors=True handles high-weight hyperedges without crashing
        return len(circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True))
    except ValueError:
        return 0

def evaluate_schedule(args):
    Hx, Hz, rounds, perm_indices, adjacency, data_qubits = args
    
    schedule = {}
    for anc_idx, targets in adjacency.items():
        # Determine schedule based on permutation of indices
        # If check has weight W and perm has length L >= W, we map priorities.
        current_perm = perm_indices[:len(targets)]
        
        # If the check weight is larger than the generated permutation (rare if max_weight used),
        # fallback to sequential for the remainder.
        
        for priority, target_index_in_list in enumerate(current_perm):
            if target_index_in_list < len(targets):
                data_idx = targets[target_index_in_list]
                schedule[(anc_idx, data_idx)] = priority

    try:
        # Generate circuit with minimal noise to define the graph
        circuit = generate_css_memory_experiment(
            Hx, Hz, rounds=rounds, memory_basis="Z", cnot_schedule=schedule
        )
        
        noisy_circuit = standard_depolarizing_noise_model(
            circuit=circuit,
            data_qubits=data_qubits,
            after_clifford_depolarization=1e-3,
            after_reset_flip_probability=1e-3,
            before_measure_flip_probability=1e-3,
            before_round_data_depolarization=1e-3
        )
        
        return get_effective_distance(noisy_circuit), schedule
    except Exception:
        return 0, schedule

# --- OPTIMIZATION ROUTINE ---

def optimize_code_schedule(n, config, max_attempts=500):
    print(f"\n--- Optimizing n={n} ---")
    try:
        Hx, Hz = get_parity_matrices(n, config)
    except Exception as e:
        print(f"Failed to load matrices for n={n}: {e}")
        return {}, 0

    num_data = Hx.shape[1]
    data_qubits = list(range(num_data))
    
    # Build Adjacency
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
    
    # --- STRATEGY GENERATION ---
    candidates = []
    
    # 1. Deterministic Heuristics (Always try these)
    # Sequential (0, 1, 2, ...)
    candidates.append(tuple(range(max_weight)))
    # Reverse (..., 2, 1, 0)
    candidates.append(tuple(range(max_weight-1, -1, -1)))
    # Interleaved (0, 2, 1, 3...) - good for coupling map conflicts
    if max_weight >= 4:
        # Creates (0, 2, 1, 3, 4, 6, 5, 7...) pattern
        pairs = []
        for i in range(0, max_weight - 1, 2):
            pairs.extend([i, i+1] if i % 4 == 0 else [i+1, i])
        if max_weight % 2 == 1:
            pairs.append(max_weight - 1)
        candidates.append(tuple(pairs))

    # 2. Fill remaining with Random or Exhaustive search
    remaining_slots = max_attempts - len(candidates)
    
    if max_weight <= 7:
        # Small weight: enumerate ALL permutations
        print(f"  Small weight ({max_weight}), generating all permutations...")
        all_perms = list(itertools.permutations(range(max_weight)))
        # Filter existing
        all_perms = [p for p in all_perms if p not in candidates]
        if len(all_perms) > remaining_slots:
            candidates.extend(random.sample(all_perms, remaining_slots))
        else:
            candidates.extend(all_perms)
    else:
        # Large weight: Random sampling to avoid memory explosion (Factorial limit)
        print(f"  Large weight ({max_weight}), using random sampling...")
        for _ in range(remaining_slots):
            p = list(range(max_weight))
            random.shuffle(p)
            candidates.append(tuple(p))

    print(f"  Evaluating {len(candidates)} schedules...")
    
    # Rounds: d+1 is sufficient to detect distance issues
    rounds = config['dist_dict'][n] + 1
    
    # Multiprocessing
    worker_args = [(Hx, Hz, rounds, p, adjacency, data_qubits) for p in candidates]
    
    best_dist = -1
    best_schedule = None
    
    # Use fewer processes to avoid memory pressure on large codes
    num_procs = min(cpu_count(), 20) 
    
    with Pool(processes=num_procs) as pool:
        # Use imap_unordered for progress visibility if needed, or simple map
        results = pool.map(evaluate_schedule, worker_args)
        
    for dist, sched in results:
        if dist > best_dist:
            best_dist = dist
            best_schedule = sched
            
    theoretical_d = config['dist_dict'][n]
    print(f"  Result: Eff_Dist={best_dist} (Theory={theoretical_d})")
    
    return best_schedule, best_dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_type", type=str, choices=["self_dual", "dual_containing"], required=True)
    parser.add_argument("--n_list", type=int, nargs='+', help="List of n values")
    parser.add_argument("--output_dir", type=str, default="schedules_cache")
    parser.add_argument("--max_attempts", type=int, default=200, help="Max schedules to test per code")
    args = parser.parse_args()

    config = CODE_CONFIGS[args.code_type]
    n_targets = args.n_list if args.n_list else config['n_list']
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    summary = []

    for n in n_targets:
        if n not in config['dist_dict']:
            print(f"Skipping n={n} (No config info)")
            continue
            
        schedule, eff_dist = optimize_code_schedule(n, config, args.max_attempts)
        
        if schedule:
            filename = f"sched_{args.code_type}_n{n}.json"
            filepath = os.path.join(args.output_dir, filename)
            
            # Save using string keys "anc,data"
            serializable = {f"{k[0]},{k[1]}": v for k, v in schedule.items()}
            
            with open(filepath, 'w') as f:
                json.dump(serializable, f)
            print(f"  Saved to {filepath}")
            summary.append((n, eff_dist, config['dist_dict'][n]))
        else:
            print(f"  Failed to generate schedule for n={n}")

    print("\n--- Summary ---")
    print(f"{'n':<6} | {'Eff. D':<8} | {'Theor. D':<8}")
    print("-" * 28)
    for n, ed, td in summary:
        print(f"{n:<6} | {ed:<8} | {td:<8}")
    
    summary_lines = []
    header = f"Optimization Summary (Code Type: {args.code_type})"
    summary_lines.append(header)
    summary_lines.append("=" * 40)
    summary_lines.append(f"{'n':<6} | {'Eff. D':<8} | {'Theor. D':<8}")
    summary_lines.append("-" * 28)
    
    for n, ed, td in summary:
        summary_lines.append(f"{n:<6} | {ed:<8} | {td:<8}")
    
    # Print to console
    print("\n")
    for line in summary_lines:
        print(line)
        
    # Write to file
    summary_file_path = os.path.join(args.output_dir, "optimization_summary.txt")
    try:
        with open(summary_file_path, "w") as f:
            f.write("\n".join(summary_lines))
            f.write("\n")
        print(f"\nSummary successfully written to: {summary_file_path}")
    except Exception as e:
        print(f"\nFailed to write summary file: {e}")

if __name__ == "__main__":
    main()