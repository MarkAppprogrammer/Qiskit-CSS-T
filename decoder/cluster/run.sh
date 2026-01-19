#!/bin/bash
#SBATCH --job-name=Stim_MWPF
#SBATCH --partition=normal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# --- CONFIGURATION ---
readonly BASE_DIR="$HOME/work/Qiskit-CSS-T/decoder"
readonly DATA_ROOT="$BASE_DIR/data"
readonly ENV_DIR="$BASE_DIR/.env"

readonly SRC_PYTHON_SCRIPT="$BASE_DIR/src/mpi_stim.py"
readonly SRC_DEP_NOISE="$BASE_DIR/src/noise_models.py"
readonly SRC_DEP_CIRCUIT="$BASE_DIR/src/circuit.py"
readonly SRC_DEP_DECODER="$BASE_DIR/src/sinter_decoders.py"
readonly SRC_DEP_CONVERT="$BASE_DIR/../doubling-CSST/convert_alist.py"
readonly SRC_ALIST_ROOT="$BASE_DIR/../doubling-CSST/alistMats"

readonly PYTHON_FILENAME="mpi_stim.py"
readonly NOISE_FILENAME="noise_models.py"
readonly CIRCUIT_FILENAME="circuit.py"      
readonly CONVERT_FILENAME="convert_alist.py"
readonly DECODER_FILENAME="sinter_decoders.py"

log_info() { printf "✅ %s\n" "$1"; }
log_error() { printf "❌ Error: %s\n" "$1" >&2; exit 1; }

submission_preflight_checks() {
    log_info "Running pre-flight checks..."
    [[ -f "$SRC_PYTHON_SCRIPT" ]] || log_error "Python script missing: $SRC_PYTHON_SCRIPT"
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

    # Copy Scripts
    cp "$SRC_PYTHON_SCRIPT" "$VERSION_DIR/$PYTHON_FILENAME"
    cp "$SRC_DEP_NOISE" "$VERSION_DIR/$NOISE_FILENAME"
    cp "$SRC_DEP_CIRCUIT" "$VERSION_DIR/$CIRCUIT_FILENAME"
    cp "$SRC_DEP_DECODER" "$VERSION_DIR/$DECODER_FILENAME"
    cp "$SRC_DEP_CONVERT" "$VERSION_DIR/$CONVERT_FILENAME" || log_info "Warning: convert_alist.py not found"
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
    
    local job_id
    job_id=$(sbatch run.sh)
    log_info "--> Submitted job ID: ${job_id##* }"
}

run_compute_logic() {
    set -e
    cd "$SLURM_SUBMIT_DIR"

    echo "--- Loading Modules ---"
    module purge
    module load gnu12/12.2.0 openmpi4/4.1.5 python/3.10.19

    # --- CRITICAL: THREADING & MEMORY CONTROL ---
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1 

    # --- ATTEMPT CLEANUP ---
    rm -f /dev/shm/sem.* 2>/dev/null || true

    # Source the LOCAL FROZEN environment
    if [[ -f "./.env/bin/activate" ]]; then
        echo "Activating local frozen environment..."
        source "./.env/bin/activate"
    else
        log_error "Could not find local environment at ./.env/bin/activate"
    fi

    echo "--- Starting Sinter Simulation ---"
    
    # 1. RUN PARALLEL
    WORKERS=$((SLURM_CPUS_PER_TASK - 4))
    if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
    
    BASE_OUTPUT="results_v${SLURM_JOB_ID}_dual.csv"
    
    echo "Running with $WORKERS workers per node..."

    srun python "$PYTHON_FILENAME" \
        --workers "$WORKERS" \
        --output "$BASE_OUTPUT" \
        --code_type "dual_containing"

    # 2. MERGE RESULTS
    echo "--- Merging CSVs ---"
    
    BASE_NAME_NO_EXT="${BASE_OUTPUT%.csv}"

    for model in "depolarizing" "si1000"; do
        FINAL_MERGED="${BASE_NAME_NO_EXT}_${model}.csv"
        FIRST_FILE=$(ls ${BASE_NAME_NO_EXT}_${model}_rank*.csv 2>/dev/null | head -n 1)
        
        if [[ -n "$FIRST_FILE" ]]; then
            head -n 1 "$FIRST_FILE" > "$FINAL_MERGED"
            for f in ${BASE_NAME_NO_EXT}_${model}_rank*.csv; do
                tail -n +2 "$f" >> "$FINAL_MERGED"
                rm "$f"
            done
            echo "✅ Merged $model data into $FINAL_MERGED"
        else
            echo "⚠️ No output files found for model: $model"
        fi
    done

    echo "--- Job Finished ---"
}

if [[ -z "$SLURM_JOB_ID" ]]; then
    run_submission_logic
else
    run_compute_logic
fi