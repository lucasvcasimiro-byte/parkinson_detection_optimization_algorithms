# Optimization Algorithms — Parkinson's Disease Detection

Population-based optimization of a Multi-Layer Perceptron (MLP) classifier for the Oxford Parkinson's Disease Detection dataset using a **Genetic Algorithm (GA)** and a **Grey Wolf Optimizer (GWO)**.

---

## Dependencies

Install all required packages with:

```bash
pip install -r requirements.txt
```

Required packages:
| Package | Version |
|---------|---------|
| `numpy` | ≥ 1.22.0 |
| `pandas` | ≥ 1.4.0 |
| `scikit-learn` | ≥ 1.0.0 |
| `scipy` | ≥ 1.9.0 |

Python **3.9+** is recommended.

---

## Project Structure

```
optimization_algorithms/
│
├── data/
│   └── parkinsons_preprocessed.csv   # Standardised Parkinson's dataset
│
├── optimization_models/
│   ├── ga.py                          # Genetic Algorithm
│   ├── ga_operators.py                # Selection, crossover & mutation operators
│   └── grey_wolf_optimizer.py         # Grey Wolf Optimizer
│
├── results/                           # Auto-generated output CSVs
│   ├── ga_operator_comparison.csv
│   ├── ga_convergence.csv
│   ├── gwo_results.csv
│   ├── gwo_convergence.csv
│   ├── statistical_comparison.csv
│   └── statistical_summary.csv
│
├── run.py                             # Main script — runs all experiments
├── utils.py                           # Shared utilities (fitness, generate_solution, …)
└── requirements.txt
```

---

## How to Run

### Run all experiments (recommended)

```bash
python run.py
```

This executes **three sequential steps**:

1. **GA Operator Comparison** — runs the GA across all combinations of initialization method × crossover operator × mutation operator and saves results to `results/ga_operator_comparison.csv` and `results/ga_convergence.csv`.

2. **GWO Comparison** — runs the Grey Wolf Optimizer with both initialization methods and saves results to `results/gwo_results.csv` and `results/gwo_convergence.csv`.

3. **Statistical Comparison** — runs GA and GWO each 10 times independently, prints mean ± std Weighted F1 scores, runs a **Mann-Whitney U test**, and saves per-run data to `results/statistical_comparison.csv`.

> ⏱ Expected total runtime: **~15–25 minutes** depending on hardware.

---

### Run individual experiments

You can import and call any function from `run.py` directly:

```python
from run import run_ga_experiment, run_gwo_experiment, run_statistical_comparison

# Single GA run with custom settings
best_solution, best_fitness, history = run_ga_experiment(
    chosen_architecture=(10,),
    pop_size=50,
    generations=100,
    verbose=True,
)

# Single GWO run
best_solution, best_fitness, history = run_gwo_experiment(
    chosen_architecture=(10,),
    pop_size=50,
    generations=100,
    verbose=True,
)

# Statistical comparison over N runs
run_statistical_comparison(num_runs=10)
```

---

## Implemented Operators (GA)

| Category | Operator | Notes |
|----------|----------|-------|
| Selection | `tournament_selection` | Selects best among k random candidates |
| Crossover | `arithmetic_crossover` | Convex blend along the line joining two parents |
| Crossover | `simulated_binary_crossover` | Polynomial spread factor, real-valued SBX |
| Mutation | `gaussian_mutation` | Additive Gaussian noise per gene |
| Mutation | `uniform_continuous_mutation` | Full gene reset to random value in bounds |
| Initialization | `uniform` (U[−1, 1]) | Both algorithms |
| Initialization | `normal` (N(0, 1)) | Both algorithms |