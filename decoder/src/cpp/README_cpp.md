# C++ Relay-BP Decoder Simulation

This directory contains the high-performance C++ implementation of the Relay-BP decoder for Quantum LDPC codes. It uses OpenMP for multi-core parallelism to accelerate large-scale simulations (up to 1M shots).

## Prerequisites

*   **Compiler**: `g++` with OpenMP support (`libomp` on macOS, usually included on Linux).
*   **Python**: Python 3 with `pandas` and `matplotlib` installed (for plotting results).

## File Structure

*   `relay_bp_decoder.h / .cpp`: The core decoder implementation.
*   `sim_relay.cpp`: The simulation driver (handles datasets, OpenMP loops, CSV writing).
*   `../plot_relay.py`: Helper script to generate plots from the CSV output.

## Compilation

Run the following command from the **root of the repository**:

```bash
g++ -O3 -fopenmp Qiskit-CSS-T/decoder/src/cpp/sim_relay.cpp Qiskit-CSS-T/decoder/src/cpp/relay_bp_decoder.cpp -o Qiskit-CSS-T/decoder/src/cpp/sim_relay
```

## Running the Simulation

Execute the binary from the **root of the repository** to ensure relative paths to `alist` files and the `fig` directory are correct.

**1. Activate Python Environment (for plotting)**
```bash
source venv/bin/activate
```

**2. Run Simulation**
```bash
Qiskit-CSS-T/decoder/src/cpp/sim_relay
```

This will:
1.  Run simulations for shot counts `[10, 100, 1k, 10k, 100k, 1M]` and biases `[0.0, 0.5]`.
2.  Save CSV datasets to `Qiskit-CSS-T/decoder/fig/RelayBP/`.
3.  Automatically call `plot_relay.py` to generate PNG plots in the same folder.

## Manual Plotting

If the automatic plotting fails during the C++ run, you can generate the plots manually:

```bash
# From repo root
for f in Qiskit-CSS-T/decoder/fig/RelayBP/*.csv; do 
    python3 Qiskit-CSS-T/decoder/src/plot_relay.py "$f"
done
```
