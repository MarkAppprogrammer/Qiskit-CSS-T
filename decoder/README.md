This project implements a circuit-level noise simulation framework for Quantum Error Correction (QEC) codes. It integrates **Stim** and **Sinter** for high-performance simulations and supports two sources of parity check matrices:

1. **Local Alist Files:** Custom codes (e.g., Quantum LDPC codes) defined in `.alist` format.
2. **Geometric Codes:** 2D and 3D topological codes (Toric, Cubic, Tetrahedral) generated dynamically via the `qcodeplot3d` package.

## 1. Installation & Requirements

Ensure you have Python 3.9+ installed.

### Python Dependencies

Install the core simulation and math libraries:

```bash
pip install numpy pandas stim sinter galois

```

### Optional/Custom Dependencies

* **qcodeplot3d:** Required if running geometric codes (Cubic, Tetrahedral, etc.). Ensure this package is in your `PYTHONPATH` or installed in your environment.
* **convert_alist:** Required if using local `.alist` files. Ensure your helper script for reading alist files is accessible.

## 2. Project Structure

* **`main.py`**: The entry point for running Monte Carlo simulations. It manages parallel workers (via Sinter) and collects results.
* **`optimize_schedule.py`**: A pre-computation script that finds the optimal CNOT ordering for your stabilizers to minimize effective error distance.
* **`circuit.py`**: Generates the Stim circuits, handling coordinate mapping and CNOT scheduling.
* **`helpers.py`**: Handles loading matrices from `.alist` files or generating them via `qcodeplot3d`.
* **`noise_models.py`**: Defines physical noise models (Standard Depolarizing, SI1000, etc.).

## 3. Usage

### A. Running Simulations (`main.py`)

The main script runs a batch of simulations for a specific code family across different physical error rates.

**Basic Command:**

```bash
python main.py --code_type <TYPE> --workers <NUM_WORKERS> --max_shots <SHOTS>

```

**Arguments:**

* `--code_type`: The key from `CODE_CONFIGS` (e.g., `self_dual`, `cubic`, `tetrahedral`).
* `--workers`: Number of CPU cores to use (defaults to all available).
* `--max_shots`: Total number of shots (decoding attempts) per error rate.
* `--output`: Output CSV filename (default: `results.csv`).

**Examples:**

1. **Run a Geometric Code (e.g., Cubic 3D code):**
```bash
python main.py --code_type cubic --output results_cubic.csv

```


*Note: For geometric codes, the system iterates over **Distance** (d=3, 5, 7...).*
2. **Run a Local Alist Code (e.g., Self-Dual QLDPC):**
```bash
python main.py --code_type self_dual --output results_qldpc.csv

```


*Note: For local codes, the system iterates over **Qubit Count** (n=8, 10, 12...) as defined in your config.*

### B. Optimizing Schedules (`optimize_schedule.py`)

Before running a large simulation, it is highly recommended to generate optimized CNOT schedules. This script tests random permutations of CNOT orders to find one that maximizes the "effective distance" of the circuit.

**Command:**

```bash
python optimize_schedule.py --code_type <TYPE> --max_attempts 500

```

**What happens?**

1. The script generates a `sched_<type>_val<X>.json` file in the `schedules_cache/` directory.
2. `main.py` automatically looks for these files. If found, it uses the optimized schedule; otherwise, it falls back to a default schedule.

**Example:**

```bash
# Optimize schedules for all Cubic codes defined in config
python optimize_schedule.py --code_type cubic

# Optimize only for specific distances
python optimize_schedule.py --code_type cubic --vals 3 5

```

## 4. Configuration

To add new codes, edit the `CODE_CONFIGS` dictionary in `main.py`.

### Adding a Local Alist Family

Set `source` to `"local"` (or omit it). You must provide a directory containing the `.alist` files and a dictionary mapping  (qubits) to  (distance).

```python
"my_new_code": {
    "source": "local",
    "dir": "./my_alist_files/",
    "dist_dict": {100: 4, 200: 6},  # Map n -> d
    "if_self_dual": True,           # If False, expects _Hx.alist and _Hz.alist
    "iter_list": [100, 200],        # List of 'n' values to simulate
    "alist_suffix": ".alist"
}

```

### Adding a Geometric Code

Set `source` to `"qcodeplot3d"`. The `iter_list` represents the **distances** () to simulate.

```python
"my_surface_code": {
    "source": "qcodeplot3d",
    "type": "square",       # Must match a case in helpers.py
    "iter_list": [3, 5, 7]  # Distances to run
}

```

## 5. Output

Results are saved to the `results/` folder (or the path specified in `--output`). The CSV contains:

* `n`: Number of physical qubits.
* `d`: Code distance.
* `p`: Physical error rate.
* `total_logical_error_rate`:  / .
* `shots`: Total number of samples.
* `noise_model`: The model used (e.g., depolarizing).

## 6. Troubleshooting

**1. `FileNotFoundError: Missing: ... .alist**`

* Ensure your `BASE_ALIST_PATH` in `main.py` points to the correct folder.
* Verify the naming convention in `helpers.py` matches your files (e.g., `n{n}_d{d}.alist`).

**2. `ImportError: qcodeplot3d package not found**`

* You are trying to run a geometric code without the generator package installed. Install it or switch to a local alist code.

**3. Simulations are slow**

* Increase `--workers`.
* Check if `sinter` is using the correct decoder (`pymatching` or `fusion_blossom` usually recommended, but this code uses `mwpf` via `SinterMWPFDecoder`). Ensure the decoder backend is optimized.

## 7. Cluster Usage (SLURM)

This project includes a robust `run.sh` script designed for SLURM-based HPC clusters. It employs a **"Snapshot & Submit"** strategy to ensure reproducibility.

### Workflow

When you execute `./run.sh` on the login node, it performs the following:

1. **Snapshot:** Creates a new versioned directory in `data/` (e.g., `data/1/`, `data/2/`).
2. **Freeze:** Copies your current python scripts, `.alist` matrices, and your **entire python virtual environment** into that directory.
3. **Submit:** Submits the job to SLURM *from inside* that frozen directory.

This guarantees that if you edit your code while a job is in the queue, the running job will not break.

### Configuration

Open `run.sh` and edit the configuration block at the top:

```bash
# Set your paths
readonly BASE_DIR="$HOME/work/my_project"
readonly ENV_DIR="$BASE_DIR/.env" # Path to your virtualenv

# Select which codes to simulate in this job
readonly CODE_TYPES_TO_RUN=("self_dual" "dual_containing")
# Or for geometric codes:
# readonly CODE_TYPES_TO_RUN=("cubic" "tetrahedral")

```

### Execution

Simply run:

```bash
./run.sh

```

### Output

* **Logs:** `data/<version>/logs/slurm_<jobid>.out`
* **Results:** `data/<version>/results_v<jobid>_<code_type>_depolarizing.csv`

### Note on Schedules

The script automatically runs `optimize_schedule.py` on the allocated head node before starting the simulation. This ensures optimized CNOT schedules are generated and cached in `data/<version>/schedules_cache/`.