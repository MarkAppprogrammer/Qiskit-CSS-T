import sys
import os
import argparse
import numpy as np
import sinter
import traceback
import time
import glob
import shutil
import uuid

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
    noise_values = np.logspace(-5, -3, 10).tolist()
    
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
        
        all_task_defs = []
        
        # 1. Generate Task Definitions
        for val in iter_list:
            try:
                Hx, Hz = get_parity_matrices(val, selected_config)
                real_n = Hx.shape[1]
                d = val if selected_config.get('source') == 'qcodeplot3d' else selected_config['dist_dict'][val]

                schedule_file = os.path.join(sched_dir, f"sched_{args.code_type}_val{val}.json")
                best_schedule = load_schedule(schedule_file) if os.path.exists(schedule_file) else {} 

                for p in noise_values:
                    circuit = generate_experiment_with_noise(
                        Hx, Hz, d*3, model_name, {"p": p}, schedule=best_schedule
                    )
                    
                    # We create the task here, but we will assign the specific decoder later
                    task_def = sinter.Task(
                        circuit=circuit, 
                        decoder='mwpf', # Placeholder
                        json_metadata={
                            'n': real_n, 'd': d, 'r': d*3, 'p': p, 
                            'noise_model': model_name, 'code_type': args.code_type, 'iter_val': val
                        }
                    )
                    all_task_defs.append(task_def)
            except Exception as e:
                if proc_rank == 0: print(f"Skipping val={val}: {e}")

        # 2. Filter tasks for THIS rank
        my_tasks = [t for i, t in enumerate(all_task_defs) if i % world_size == proc_rank]
        if not my_tasks: continue

        # 3. Setup Custom Decoders for Isolation
        custom_decoders = {}
        
        for task in my_tasks:
            meta = task.json_metadata
            # Create a unique key for this specific task
            unique_id = f"mwpf_n{meta['n']}_p{meta['p']:.6e}_{uuid.uuid4().hex[:6]}"
            
            # Create a unique trace file for this task
            trace_filename = os.path.join(tmp_dir, f"trace_{unique_id}.bin")
            
            # Clean up old file if it exists (paranoid check)
            for f in glob.glob(f"{trace_filename}*"):
                try: os.remove(f)
                except: pass

            # Update the task to use this unique decoder key
            task.decoder = unique_id
            
            # Save the trace path in metadata so the parser knows where to look later
            task.json_metadata['trace_path'] = trace_filename
            
            # Register the decoder
            custom_decoders[unique_id] = SinterMWPFDecoder(
                cluster_node_limit=50, 
                trace_filename=trace_filename
            )

        # 4. Run Sinter (Batch Mode)
        resume_path = os.path.join(tmp_dir, f"resume_{base_filename}_{model_name}_{proc_rank}.sinter")
        part_filename = os.path.join(results_dir, f"{base_filename}_{model_name}_rank{proc_rank}{output_ext}")
        
        existing_data = []
        if os.path.exists(resume_path):
            try: existing_data = sinter.stats_from_csv_files(resume_path)
            except: pass

        try:
            iterator = sinter.iter_collect(
                num_workers=args.workers, 
                tasks=my_tasks, 
                custom_decoders=custom_decoders,
                max_shots=args.max_shots,
                additional_existing_data=existing_data,
                max_batch_seconds=30,
                max_batch_size=1000
            )
            
            with open(resume_path, 'a') as resume_file:
                if resume_file.tell() == 0: print(sinter.CSV_HEADER, file=resume_file)
                
                last_print_time = 0
                last_save_time = 0
                
                for progress in iterator:
                    # 1. Save Raw Data
                    for stat in progress.new_stats:
                        print(stat.to_csv_line(), file=resume_file, flush=True)
                    
                    current_time = time.time()
                    
                    # 2. Print Live Table
                    if current_time - last_print_time > 0.5:
                        if is_slurm_mode:
                            if current_time - last_print_time > 5.0:
                                print(f"\n{progress.status_message}", flush=True)
                                last_print_time = current_time
                        else:
                            # Standard Sinter table output
                            print(f"\n{progress.status_message}", flush=True)
                            last_print_time = current_time

                    # 3. Auto-save Parsed Results
                    if current_time - last_save_time > 300.0:
                        try:
                            resume_file.flush()
                            curr_stats = sinter.stats_from_csv_files(resume_path)
                            curr_df = parse_and_average_stats(curr_stats, model_name) 
                            curr_df.to_csv(part_filename, index=False)
                        except Exception: pass
                        last_save_time = current_time

            # Final Save
            full_stats = sinter.stats_from_csv_files(resume_path)
            final_df = parse_and_average_stats(full_stats, model_name)
            final_df.to_csv(part_filename, index=False)
            
            # Cleanup Trace Files
            for t in my_tasks:
                tf = t.json_metadata.get('trace_path')
                if tf:
                    for f in glob.glob(f"{tf}*"):
                        try: os.remove(f)
                        except: pass

            print(f"[Node {proc_rank}] ✅ Finished {model_name}.")

        except Exception as e: 
            print(f"[Node {proc_rank}] Fail: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()