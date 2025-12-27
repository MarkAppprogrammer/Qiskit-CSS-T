#!/bin/bash
#SBATCH --job-name=Stim_MWPF
#SBATCH --partition=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# --- CONFIGURATION ---
# Base directories
readonly BASE_DIR="$HOME/work/Qiskit-CSS-T/decoder"
readonly DATA_ROOT="$BASE_DIR/data"

# Source File Paths
readonly SRC_PYTHON_SCRIPT="$BASE_DIR/src/mpi_stim.py"
readonly SRC_DEP_NOISE="$BASE_DIR/src/noise_models.py"
readonly SRC_DEP_CIRCUIT="$BASE_DIR/src/circuit.py"
readonly SRC_DEP_CONVERT="$BASE_DIR/../doubling-CSST/convert_alist.py"

# CHANGE 1: Point to the PARENT alist directory, not the subdirectory
readonly SRC_ALIST_ROOT="$BASE_DIR/../doubling-CSST/alistMats"

readonly PYTHON_FILENAME="mpi_stim.py"
readonly NOISE_FILENAME="noise_models.py"
readonly CIRCUIT_FILENAME="circuit.py"      
readonly CONVERT_FILENAME="convert_alist.py"

log_info() { printf "✅ %s\n" "$1"; }
log_error() { printf "❌ Error: %s\n" "$1" >&2; exit 1; }

# --- Submission Mode Functions ---
submission_preflight_checks() {
    log_info "Running pre-flight checks..."
    [[ -f "$SRC_PYTHON_SCRIPT" ]] || log_error "Python script not found at: $SRC_PYTHON_SCRIPT"
    [[ -f "$SRC_DEP_NOISE" ]] || log_error "Noise dependency not found at: $SRC_DEP_NOISE"
    [[ -f "$SRC_DEP_CIRCUIT" ]] || log_error "Circuit dependency not found at: $SRC_DEP_CIRCUIT"
    # CHANGE 2: Check the root alist directory
    [[ -d "$SRC_ALIST_ROOT" ]] || log_error "Alist root directory not found at: $SRC_ALIST_ROOT"
    log_info "Pre-flight checks passed."
}

create_version_directory() {
    log_info "Creating new version directory..."
    local latest_version
    latest_version=$(ls -1 "$DATA_ROOT" 2>/dev/null | grep '^[0-9][0-9]*$' | sort -n | tail -1)
    local new_version_num=$(( ${latest_version:-0} + 1 ))

    VERSION_DIR="$DATA_ROOT/$new_version_num"
    mkdir -p "$VERSION_DIR/logs/"
    
    # CHANGE 3: We do NOT mkdir alistMats manually, cp -r will handle it

    cp "$SRC_PYTHON_SCRIPT" "$VERSION_DIR/$PYTHON_FILENAME"
    cp "$SRC_DEP_NOISE" "$VERSION_DIR/$NOISE_FILENAME"
    cp "$SRC_DEP_CIRCUIT" "$VERSION_DIR/$CIRCUIT_FILENAME"
    cp "$SRC_DEP_CONVERT" "$VERSION_DIR/$CONVERT_FILENAME" || log_info "Warning: convert_alist.py not found, skipping."
    cp "$0" "$VERSION_DIR/run.sh"

    # CHANGE 4: Copy the entire alistMats folder recursively
    # This ensures "GO03_self_dual" and "QR_dual_containing" subfolders exist inside the destination
    cp -r "$SRC_ALIST_ROOT" "$VERSION_DIR/alistMats"

    log_info "Created new version directory: $VERSION_DIR"
}

submit_job() {
    log_info "Submitting job from $VERSION_DIR..."
    cd "$VERSION_DIR" || log_error "Could not change to directory $VERSION_DIR"

    # Default logic: Submit two jobs (one for each code type) or just one generic
    # For now, let's just submit one job that runs the default (or you can edit args here)
    local job_id
    job_id=$(sbatch --output="logs/slurm_%j.out" \
                    --error="logs/slurm_%j.err" \
                    run.sh)

    log_info "--> Submitted job ID: ${job_id##* }"
}

run_submission_logic() {
    submission_preflight_checks
    create_version_directory
    submit_job
}

# --- Compute Mode Function ---
run_compute_logic() {
    set -e
    cd "$SLURM_SUBMIT_DIR"

    echo "--- Loading Modules ---"
    module purge
    module load gnu12/12.2.0 openmpi4/4.1.5 python/3.10.19

    if [[ -f "$BASE_DIR/.env/bin/activate" ]]; then
        source "$BASE_DIR/.env/bin/activate"
    fi

    echo "--- Starting Distributed Sinter Simulation ---"
    echo "Nodes: $SLURM_NNODES"
    echo "Cores per Node: $SLURM_CPUS_PER_TASK"
    
    # 1. RUN PARALLEL (You can change --code_type here if needed, or pass it as an arg to sbatch)
    # Defaulting to dual_containing as requested in previous contexts
    
    echo "Running Dual Containing..."
    srun python "$PYTHON_FILENAME" \
        --workers "$SLURM_CPUS_PER_TASK" \
        --output "results_v${SLURM_JOB_ID}_dual.csv" \
        --code_type "dual_containing"

    # Uncomment below to run self_dual as well in the same job, 
    # OR submit a separate job with --code_type self_dual
    
    # echo "Running Self Dual..."
    # srun python "$PYTHON_FILENAME" \
    #     --workers "$SLURM_CPUS_PER_TASK" \
    #     --output "results_v${SLURM_JOB_ID}_self.csv" \
    #     --code_type "self_dual"

    # 2. MERGE RESULTS (Simple merge for the dual_containing run)
    echo "--- Merging CSVs ---"
    
    for model in "depolarizing" "si1000"; do
        OUT_FILE="results_v${SLURM_JOB_ID}_dual_${model}.csv"
        
        # Merge only if files exist
        FIRST_FILE=$(ls results_v${SLURM_JOB_ID}_dual_${model}_rank*.csv 2>/dev/null | head -n 1)
        
        if [[ -n "$FIRST_FILE" ]]; then
            head -n 1 "$FIRST_FILE" > "$OUT_FILE"
            for f in results_v${SLURM_JOB_ID}_dual_${model}_rank*.csv; do
                tail -n +2 "$f" >> "$OUT_FILE"
                rm "$f" 
            done
            echo "Merged $model results into $OUT_FILE"
        fi
    done

    echo "--- Job Finished ---"
}

# --- Entry Point ---
if [[ -z "$SLURM_JOB_ID" ]]; then
    run_submission_logic
else
    run_compute_logic
fi