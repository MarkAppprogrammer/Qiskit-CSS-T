import sys
import os
import argparse
import numpy as np
import sinter
import traceback
import time
import glob

# Add local directory
sys.path.append(".")

from helpers import get_parity_matrices, parse_and_average_stats, CODE_CONFIGS
from circuit import generate_experiment_with_noise, load_schedule
from mwpf.sinter_decoders import SinterMWPFDecoder 

def main():
    parser = argparse.ArgumentParser()
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    is_slurm_mode = slurm_cpus is not None
    
    default_workers = int(slurm_cpus) if is_slurm_mode else os.cpu_count()
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--max_shots", type=int, default=1_000_000)
    parser.add_argument("--output", type=str, default="results.csv")
    parser.add_argument("--code_type", type=str, required=True, choices=CODE_CONFIGS.keys())
    args = parser.parse_args()

    proc_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    selected_config = CODE_CONFIGS[args.code_type]
    iter_list = selected_config.get("iter_list", selected_config.get("n_list", [])) 
    noise_values = np.logspace(-5, -2, 15).tolist()
    
    ver_dir = os.path.dirname(args.output)
    if not ver_dir: ver_dir = "."
    results_dir = os.path.join(ver_dir, "results")
    tmp_dir = os.path.join(ver_dir, "tmp")
    sched_dir = os.path.join(ver_dir, "schedules_cache")
    
    if proc_rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(sched_dir, exist_ok=True)
    time.sleep(2)

    base_filename = os.path.splitext(os.path.basename(args.output))[0]
    output_ext = os.path.splitext(args.output)[1]

    for model_name in ["depolarizing"]:
        all_tasks = []
        for val in iter_list:
            try:
                # 1. Load Parity Matrices
                Hx, Hz = get_parity_matrices(val, selected_config)
                
                # 2. Determine n and d
                real_n = Hx.shape[1]
                if selected_config.get('source') == 'qcodeplot3d':
                    d = val
                else:
                    d = selected_config['dist_dict'][val]

                # 3. Load Schedule
                schedule_file = os.path.join(sched_dir, f"sched_{args.code_type}_val{val}.json")
                best_schedule = None
                if os.path.exists(schedule_file):
                    if proc_rank == 0: print(f"[Node 0] Loading schedule for {args.code_type} val={val}")
                    best_schedule = load_schedule(schedule_file)
                else:
                    if proc_rank == 0: print(f"[Node 0] Default schedule for {args.code_type} val={val}")
                    best_schedule = {} 

                # 4. Generate Tasks
                for p in noise_values:
                    circuit = generate_experiment_with_noise(
                        Hx, Hz, d*3, model_name, {"p": p}, schedule=best_schedule
                    )
                    all_tasks.append(sinter.Task(
                        circuit=circuit, decoder='mwpf',
                        json_metadata={
                            'n': real_n, 
                            'd': d, 
                            'r': d*3, 
                            'p': p, 
                            'noise_model': model_name, 
                            'code_type': args.code_type,
                            'iter_val': val
                        }
                    ))
            except Exception as e: 
                print(f"[Node {proc_rank}] Skipping val={val}: {e}")
                # traceback.print_exc()

        my_tasks = [t for i, t in enumerate(all_tasks) if i % world_size == proc_rank]
        if not my_tasks: continue
        
        resume_path = os.path.join(tmp_dir, f"resume_{base_filename}_{model_name}_{proc_rank}.sinter")
        trace_path = os.path.join(tmp_dir, f"trace_{base_filename}_{model_name}_{proc_rank}.bin")
        part_filename = os.path.join(results_dir, f"{base_filename}_{model_name}_rank{proc_rank}{output_ext}")

        existing_data = []
        if os.path.exists(resume_path):
            try: existing_data = sinter.stats_from_csv_files(resume_path)
            except Exception: pass
        else:
            for old_f in glob.glob(f"{trace_path}*"):
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
            
            last_print_time = 0
            last_disk_write_time = 0 
            
            with open(resume_path, 'a') as resume_file:
                if resume_file.tell() == 0: print(sinter.CSV_HEADER, file=resume_file)
                
                for progress in iterator:
                    # 1. Write Raw Data
                    for stat in progress.new_stats:
                        print(stat.to_csv_line(), file=resume_file, flush=True)
                    
                    current_time = time.time()
                    
                    # 2. Print Progress to Console
                    if current_time - last_print_time > 5.0: 
                        if is_slurm_mode:
                            if proc_rank == 0:
                                print(f"[Node 0] Progress: {progress.status_message}", flush=True)
                        else:
                            # Use carriage return \r to update the same line locally
                            print(f"\r{progress.status_message}", end="", flush=True)
                        last_print_time = current_time

                    # 3. Auto-Save Final CSV
                    if current_time - last_disk_write_time > 300.0:
                        try:
                            resume_file.flush() 
                            current_full_stats = sinter.stats_from_csv_files(resume_path)
                            current_df = parse_and_average_stats(current_full_stats, trace_path, model_name)
                            current_df.to_csv(part_filename, index=False)
                            if proc_rank == 0:
                                if is_slurm_mode:
                                    print(f"\n[Node 0] (Auto-Save) Updated {part_filename}", flush=True)
                        except Exception as e:
                            pass # Ignore write errors during runtime
                        
                        last_disk_write_time = current_time
            
            if not is_slurm_mode: print() # Newline after progress bar
            
            # Final Save
            full_stats = sinter.stats_from_csv_files(resume_path)
            final_df = parse_and_average_stats(full_stats, trace_path, model_name)
            final_df.to_csv(part_filename, index=False)
            print(f"[Node {proc_rank}] ✅ Finished {model_name}.")
            
        except Exception as e: 
            print(f"[Node {proc_rank}] Fail: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()