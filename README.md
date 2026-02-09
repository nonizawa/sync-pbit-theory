# Oscillation Suppression in Synchronous p-bit Updates

This repository contains the simulation and theory scripts used in the manuscript
on oscillation suppression under tick-random synchronous updates.

## Contents
- `theory/`: finite-time stability calculations and critical boundary computation.
- `sim/`: p-bit simulation and data generation utilities.

## Requirements
- Python 3.10+
- numpy
- scipy
- matplotlib
- pandas
- certifi (optional, for HTTPS downloads)

Install with:
```bash
pip install -r requirements.txt
```

## Data
The theory scripts expect G-set graphs under `./gset`.
If you do not have G-set files, download them and place the `.txt` files in
`github/gset/` before running the theory scripts.

## Usage
Theory (critical boundary):
```bash
python theory/theory_cstar_6graphs.py
```

Theory with IPR correction:
```bash
python theory/theory_transient_6graphs_ipr.py
```

Simulation utilities:
```bash
python sim/pbit_doe.py --help
```

## Outputs
Scripts write plots and CSVs to subfolders such as:
- `theory_transient_6graphs/`
- `energy_logs/`

These outputs are excluded by `.gitignore`.

