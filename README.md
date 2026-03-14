# RL-QC: Quantum Circuit Mapping Experiments

This project explores strategies for mapping logical qubits to physical qubits on noisy quantum devices. It evaluates the performance of various mapping algorithms under realistic noise models using Qiskit's fake provider backends.

## Features

- **Multiple Mapping Strategies**:
  - Random mapping (baseline)
  - Calibration-based best/worst mappings
  - Murali-inspired reliability optimization
  - Tannu-VQA-like connectivity-based mapping

- **Circuit Benchmarks**:
  - Bell state preparation
  - GHZ state (3 qubits)
  - Adder circuit (quantum ripple carry adder)

- **Noise-Aware Evaluation**:
  - Simulates readout and two-qubit gate errors
  - Computes success probabilities via Monte Carlo sampling
  - Generates hardware mapping visualizations

- **Extensible Framework**:
  - Easy to add new circuits and mapping strategies
  - Supports various IBM Quantum fake backends

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rl-qc
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Usage

Run the main experiment script:

```bash
uv run python src/noise_mapping_exp.py [options]
```

### Command-Line Options

- `--backend`: Fake backend to use (default: `FakeManilaV2`)
  - Available: `FakeManilaV2`, `FakeLimaV2`, `FakeQuitoV2`, `FakeBogotaV2`, `FakeRomeV2`
- `--shots`: Number of measurement shots (default: 4096)
- `--seed`: Random seed for reproducibility (default: 0)
- `--random_trials`: Number of random mapping trials (default: 10)
- `--omega`: Weight for readout vs. gate errors in calibration scoring (default: 0.5)
- `--out-dir`: Directory for output visualizations (default: `mapping_visualizations/`)

### Example

```bash
uv run python src/noise_mapping_exp.py --backend FakeManilaV2 --shots 4096 --seed 42
```

This will:
- Run experiments on all benchmark circuits
- Compare mapping strategies
- Generate mapping visualization images
- Print success probabilities and layouts

## Output

The script generates:
- Console output with success rates for each mapping strategy and circuit
- PNG visualization files showing qubit mappings on hardware topology
- Bar chart comparing success probabilities across methods

## Project Structure

```
rl-qc/
├── pyproject.toml          # Project configuration and dependencies
├── README.md              # This file
├── src/
│   └── noise_mapping_exp.py  # Main experiment script
└── mapping_visualizations/  # Output directory for plots
```

## Dependencies

Key dependencies (managed via `pyproject.toml`):
- `qiskit`: Quantum circuit simulation
- `qiskit-aer`: High-performance simulator
- `qiskit-ibm-runtime`: Access to fake backends
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualizations
- `networkx`: Graph algorithms for hardware topology
- `gymnasium`, `stable-baselines3`: For future RL components

## Authors

Ryan Luedtke

## Acknowledgments

This work builds upon research in quantum circuit compilation and noise-aware mapping strategies, including methods inspired by Murali et al. and Tannu et al.