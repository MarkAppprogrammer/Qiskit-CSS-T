#!/bin/bash
#SBATCH --job-name=Stim_MWPF
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/restart_%j.out
#SBATCH --error=logs/restart_%j.err

# --- CONFIGURATION ---
# 1. Define your base paths
readonly BASE_DIR="$HOME/work/Qiskit-CSS-T/decoder"
readonly SRC_DIR="$BASE_DIR/src"
readonly DATA_ROOT="$BASE_DIR/data"

# 2. Determine which directory to restart
# If TARGET_DIR is passed via --export, use it. Otherwise, find the latest version.
if [[ -z "$TARGET_DIR" ]]; then
    LATEST_VER=$(ls -1 "$DATA_ROOT" 2>/dev/null | grep '^[0-9][0-9]*$' | sort -n | tail -1)
    if [[ -z "$LATEST_VER" ]]; then
        echo "❌ Error: Could not find any version directories in $DATA_ROOT" >&2
        exit 1
    fi
    TARGET_DIR="$DATA_ROOT/$LATEST_VER"
    echo "⚠️  No TARGET_DIR provided. Defaulting to latest: $TARGET_DIR"
else
    echo "🎯 Restarting user-specified target: $TARGET_DIR"
fi

# 3. Check validity
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "❌ Error: Target directory does not exist: $TARGET_DIR" >&2
    exit 1
fi

# --- EXECUTION LOGIC ---
set -e # Exit immediately on error

echo "--- Preparing Restart Environment ---"

# A. Move to the target directory
cd "$TARGET_DIR"
echo "Working directory: $(pwd)"

# B. CRITICAL: Update the python scripts from SRC
echo "Updating scripts from $SRC_DIR..."
cp "$SRC_DIR/mpi_stim.py" .
cp "$SRC_DIR/noise_models.py" .
cp "$SRC_DIR/circuit.py" .

# C. Load Environment
module purge
module load gnu12/12.2.0 openmpi4/4.1.5 python/3.10.19

# D. Threading Controls (The Fix for [Errno 28])
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 

# E. Cleanup Stale Locks
rm -f /dev/shm/sem.* 2>/dev/null || true

# F. Activate Frozen Environment
if [[ -f "./.env/bin/activate" ]]; then
    echo "Activating local frozen environment..."
    source "./.env/bin/activate"
else
    echo "❌ Error: Local environment .env/bin/activate not found!" >&2
    exit 1
fi

# G. Calculate Workers
WORKERS=$((SLURM_CPUS_PER_TASK - 4))
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi

echo "--- Starting Resumed Simulation ---"
echo "Running with $WORKERS workers per node..."

# Ensure results/ directory exists
mkdir -p results/

# --- H. RUN LOOP ---
# We re-run both code types. Sinter generates new tasks, so this is safe.
# If you only want to run one, you can comment the loop out.
for CODE_TYPE in "self_dual" "dual_containing"; do
    
    echo "========================================"
    echo "Resuming Code Type: $CODE_TYPE"
    echo "========================================"

    # Note: New Job ID means new output files. 
    # We do NOT overwrite the old crashed files, which is safer.
    BASE_OUTPUT="results_v${SLURM_JOB_ID}_${CODE_TYPE}.csv"

    # 1. RUN PARALLEL
    srun python "mpi_stim.py" \
        --workers "$WORKERS" \
        --output "$BASE_OUTPUT" \
        --code_type "$CODE_TYPE"

    # 2. MERGE RESULTS
    echo "--- Merging CSVs for $CODE_TYPE ---"
    
    BASE_NAME_NO_EXT="${BASE_OUTPUT%.csv}"

    for model in "depolarizing" "si1000"; do
        FINAL_MERGED="${BASE_NAME_NO_EXT}_${model}.csv"
        
        # Look in results/ subdirectory where workers write shards
        FIRST_FILE=$(ls results/${BASE_NAME_NO_EXT}_${model}_rank*.csv 2>/dev/null | head -n 1)
        
        if [[ -n "$FIRST_FILE" ]]; then
            head -n 1 "$FIRST_FILE" > "$FINAL_MERGED"
            for f in results/${BASE_NAME_NO_EXT}_${model}_rank*.csv; do
                tail -n +2 "$f" >> "$FINAL_MERGED"
                rm "$f" 
            done
            echo "✅ Merged $model data into $FINAL_MERGED"
        else
            echo "⚠️ No output files found for model: $model"
        fi
    done
done

echo "--- Restart Complete ---"