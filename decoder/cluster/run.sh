#!/bin/bash
#SBATCH --job-name=Stim_MWPF
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64

# --- CONFIGURATION ---
# Base directories
readonly BASE_DIR="$HOME/work/Qiskit-CSS-T/decoder"
readonly DATA_ROOT="$BASE_DIR/data"

# Source File Paths
readonly SRC_PYTHON_SCRIPT="$BASE_DIR/src/mpi_stim.py"
readonly SRC_DEP_NOISE="$BASE_DIR/src/noise_models.py"
readonly SRC_DEP_CIRCUIT="$BASE_DIR/src/circuit.py"
readonly SRC_DEP_CONVERT="$BASE_DIR/../doubling-CSST/convert_alist.py"
readonly SRC_ALIST_DIR="$BASE_DIR/../doubling-CSST/alistMats/QR_dual_containing/"

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
    [[ -d "$SRC_ALIST_DIR" ]] || log_error "Alist directory not found at: $SRC_ALIST_DIR"
    log_info "Pre-flight checks passed."
}

create_version_directory() {
    log_info "Creating new version directory..."
    local latest_version
    latest_version=$(ls -1 "$DATA_ROOT" 2>/dev/null | grep '^[0-9][0-9]*$' | sort -n | tail -1)
    local new_version_num=$(( ${latest_version:-0} + 1 ))

    VERSION_DIR="$DATA_ROOT/$new_version_num"
    mkdir -p "$VERSION_DIR/logs/"
    mkdir -p "$VERSION_DIR/alistMats/"

    cp "$SRC_PYTHON_SCRIPT" "$VERSION_DIR/$PYTHON_FILENAME"
    cp "$SRC_DEP_NOISE" "$VERSION_DIR/$NOISE_FILENAME"
    cp "$SRC_DEP_CIRCUIT" "$VERSION_DIR/$CIRCUIT_FILENAME"
    cp "$SRC_DEP_CONVERT" "$VERSION_DIR/$CONVERT_FILENAME" || log_info "Warning: convert_alist.py not found, skipping."
    cp "$0" "$VERSION_DIR/run.sh"
    cp -r "$SRC_ALIST_DIR"/* "$VERSION_DIR/alistMats/"

    log_info "Created new version directory: $VERSION_DIR"
}

submit_job() {
    log_info "Submitting job from $VERSION_DIR..."
    cd "$VERSION_DIR" || log_error "Could not change to directory $VERSION_DIR"

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
    module load gnu12/12.2.0 openmpi/4.1.5 python/3.10.19


    if [[ -f "$BASE_DIR/.env/bin/activate" ]]; then
        echo "Activating virtual environment..."
        source "$BASE_DIR/.env/bin/activate"
    else
        echo "WARNING: Could not find virtual environment at $BASE_DIR/.env"
        echo "Ensure you ran: python3 -m venv .env"
        exit 1
    fi

    echo "--- Starting Sinter Simulation ---"
    echo "Running on $(hostname) with $SLURM_CPUS_PER_TASK CPUs"
    echo "Working directory: $PWD"

    # Run the LOCAL copy of the script
    python "$PYTHON_FILENAME" \
        --workers "$SLURM_CPUS_PER_TASK" \
        --output "results_v${SLURM_JOB_ID}.csv"

    echo "--- Job Finished ---"
}

# --- Entry Point ---
if [[ -z "$SLURM_JOB_ID" ]]; then
    run_submission_logic
else
    run_compute_logic
fi