#!/bin/bash
#SBATCH --job-name=Stim_Integrated
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# --- CONFIGURATION ---
# Define the root of your source code (where main.py lives)
readonly BASE_DIR="$HOME/work/Qiskit-CSS-T/decoder"
readonly DATA_ROOT="$BASE_DIR/data"
readonly ENV_DIR="$BASE_DIR/.env"

# Source Files Map
readonly SRC_MAIN="$BASE_DIR/src/main.py"
readonly SRC_HELPERS="$BASE_DIR/src/helpers.py"
readonly SRC_CIRCUIT="$BASE_DIR/src/circuit.py"
readonly SRC_NOISE="$BASE_DIR/src/noise_models.py"
readonly SRC_OPTIMIZE="$BASE_DIR/src/optimize_schedule.py"
# If you still use convert_alist, keep it; otherwise optional
readonly SRC_CONVERT="$BASE_DIR/../doubling-CSST/convert_alist.py" 
readonly SRC_ALIST_ROOT="$BASE_DIR/../doubling-CSST/alistMats"

# Which codes to run in this batch?
# Options: "self_dual", "dual_containing", "cubic", "tetrahedral", "square", "triangular"
readonly CODE_TYPES_TO_RUN=("self_dual" "dual_containing")

log_info() { printf "✅ %s\n" "$1"; }
log_error() { printf "❌ Error: %s\n" "$1" >&2; exit 1; }

submission_preflight_checks() {
    log_info "Running pre-flight checks..."
    [[ -f "$SRC_MAIN" ]] || log_error "Main script missing: $SRC_MAIN"
    [[ -f "$SRC_HELPERS" ]] || log_error "Helpers script missing: $SRC_HELPERS"
    [[ -d "$SRC_ALIST_ROOT" ]] || log_error "Alist directory missing: $SRC_ALIST_ROOT"
    [[ -d "$ENV_DIR" ]] || log_error "Environment missing: $ENV_DIR"
}

create_version_directory() {
    log_info "Creating new version directory..."
    local latest_version
    latest_version=$(ls -1 "$DATA_ROOT" 2>/dev/null | grep '^[0-9][0-9]*$' | sort -n | tail -1)
    local new_version_num=$(( ${latest_version:-0} + 1 ))

    VERSION_DIR="$DATA_ROOT/$new_version_num"
    mkdir -p "$VERSION_DIR/logs/"
    mkdir -p "$VERSION_DIR/schedules_cache/"

    # Copy Scripts
    cp "$SRC_MAIN" "$VERSION_DIR/main.py"
    cp "$SRC_HELPERS" "$VERSION_DIR/helpers.py"
    cp "$SRC_CIRCUIT" "$VERSION_DIR/circuit.py"
    cp "$SRC_NOISE" "$VERSION_DIR/noise_models.py"
    cp "$SRC_OPTIMIZE" "$VERSION_DIR/optimize_schedule.py"
    
    if [[ -f "$SRC_CONVERT" ]]; then
        cp "$SRC_CONVERT" "$VERSION_DIR/convert_alist.py"
    fi

    cp "$0" "$VERSION_DIR/run.sh"
    
    # Copy Assets
    cp -r "$SRC_ALIST_ROOT" "$VERSION_DIR/alistMats"
    
    # Copy Environment (Frozen for reproducibility)
    log_info "Freezing python environment (this may take a moment)..."
    cp -r "$ENV_DIR" "$VERSION_DIR/.env"

    log_info "Created version: $VERSION_DIR"
}

run_submission_logic() {
    submission_preflight_checks
    create_version_directory
    cd "$VERSION_DIR" || log_error "Failed to cd to $VERSION_DIR"
    
    # Submit from inside the version directory so output logs go there
    local job_id
    job_id=$(sbatch run.sh)
    log_info "--> Submitted job ID: ${job_id##* }"
}

run_compute_logic() {
    set -e
    cd "$SLURM_SUBMIT_DIR"

    echo "--- Loading Modules ---"
    # Adjust these to your specific cluster modules
    module purge
    module load gnu12/12.2.0 openmpi4/4.1.5 python/3.10.19

    # --- THREADING & MEMORY CONTROL ---
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1 

    # --- CLEANUP ---
    rm -f /dev/shm/sem.* 2>/dev/null || true

    # Source the LOCAL FROZEN environment
    if [[ -f "./.env/bin/activate" ]]; then
        echo "Activating local frozen environment..."
        source "./.env/bin/activate"
    else
        log_error "Could not find local environment at ./.env/bin/activate"
    fi

    # Calculate workers (reserve a few cores for overhead)
    WORKERS=$((SLURM_CPUS_PER_TASK - 4))
    if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi

    echo "--- Starting Sinter Simulation ---"
    echo "Running with $WORKERS workers per node..."

    # Ensure results dir exists
    mkdir -p results/

    # --- LOOP OVER CONFIGURED CODE TYPES ---
    for CODE_TYPE in "${CODE_TYPES_TO_RUN[@]}"; do
        
        echo "========================================"
        echo "Processing Code Type: $CODE_TYPE"
        echo "========================================"

        # 1. OPTIMIZE SCHEDULE (Run once on head node)
        echo "--> Generating/Checking Schedules..."
        python optimize_schedule.py \
            --code_type "$CODE_TYPE" \
            --output_dir "schedules_cache" \
            --max_attempts 200

        BASE_OUTPUT="results_v${SLURM_JOB_ID}_${CODE_TYPE}.csv"

        # 2. RUN PARALLEL SIMULATION
        echo "--> Launching MPI Sinter..."
        srun python main.py \
            --workers "$WORKERS" \
            --output "$BASE_OUTPUT" \
            --code_type "$CODE_TYPE"

        # 3. MERGE RESULTS
        echo "--> Merging CSV shards..."
        
        # Base name without extension (e.g. results_v123_self_dual)
        BASE_NAME_NO_EXT="${BASE_OUTPUT%.csv}"

        # Sinter/Main outputs shards like: results/{BASE}_{model}_rank{i}.csv
        # We want to merge them into one file per noise model
        
        for model in "depolarizing" "si1000"; do
            FINAL_MERGED="${BASE_NAME_NO_EXT}_${model}.csv"
            
            # Find the first available shard to grab the header
            FIRST_FILE=$(ls results/${BASE_NAME_NO_EXT}_${model}_rank*.csv 2>/dev/null | head -n 1)
            
            if [[ -n "$FIRST_FILE" ]]; then
                # Write header
                head -n 1 "$FIRST_FILE" > "$FINAL_MERGED"
                
                # Append content (skipping header) from all shards
                for f in results/${BASE_NAME_NO_EXT}_${model}_rank*.csv; do
                    tail -n +2 "$f" >> "$FINAL_MERGED"
                done
                echo "   ✅ Saved: $FINAL_MERGED"
            else
                # It's possible only one model was run, so silence this if expected
                echo "   ⚠️ No output found for model: $model (skipping)"
            fi
        done
        echo "Finished $CODE_TYPE"
        echo ""
    done

    echo "--- Job Finished ---"
}

# --- ENTRY POINT ---
if [[ -z "$SLURM_JOB_ID" ]]; then
    run_submission_logic
else
    run_compute_logic
fi