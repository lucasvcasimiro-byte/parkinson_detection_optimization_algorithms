# Optimization Algorithms — Parkinson's Disease Detection

Population-based optimization of a Multi-Layer Perceptron (MLP) classifier for the Oxford Parkinson's Disease Detection dataset using a **Genetic Algorithm (GA)** and a **Grey Wolf Optimizer (GWO)**.

Note: Consider that the README.md was created for the GitHub workflow. With only the files, the project root directory is simply the name of the folder.
---

## Dependencies

Install all required packages with:

```bash
pip install -r requirements.txt
```

| Package | Version |

|---|---|
| `numpy` | ≥ 1.22.0 |
| `pandas` | ≥ 1.4.0 |
| `scikit-learn` | ≥ 1.0.0 |
| `scipy` | ≥ 1.9.0 |
| `matplotlib` | ≥ 3.5.0 |
| `seaborn` | ≥ 0.11.0 |

---

## Project Structure

```bash
parkinson_detection_optimization_algorithms/
│
├── data/
│   └── parkinsons_preprocessed.csv       # Standardised Parkinson's dataset
│
├── optimization_models/
│   ├── __init__.py
│   ├── ga.py                              # Genetic Algorithm
│   ├── ga_operators.py                    # Selection, crossover & mutation operators
│   └── grey_wolf_optimizer.py             # Grey Wolf Optimizer
│
├── main/
│   ├── __init__.py
│   ├── utils.py                           # Shared utilities (fitness_function, generate_solution, …)
│   ├── eda.py                             # Class distribution analysis
│   ├── grid_search.py                     # GA operator & architecture grid search
│   ├── compare.py                         # GA vs GWO comparison (30 runs + statistical test)
│   ├── grid_search_visualizations.py      # Plots from grid search results
│   └── compare_visualizations.py          # Plots from GA vs GWO comparison results
│
├── results/
│   ├── csv/                               # Auto-generated output CSVs
│   │   ├── ga_grid_search.csv
│   │   └── ga_vs_gwo.csv
│   └── visualizations/                    # Auto-generated plots (.png)
│
├── requirements.txt
└── README.md
```

> **Important:** all scripts must be run from the **project root directory** using the `-m` flag, as shown below.

---

## How to Run

### Step 1 — Exploratory Data Analysis

Generates the class distribution plot to understand dataset balance before optimization.

```bash
python -m main.eda
```

Produces:

- `results/visualizations/class_distribution.png` — dataset class balance

---

### Step 2 — GA Operator Grid Search *(long, don't run again)*

Evaluates all combinations of architecture × initialization method × selection × crossover × mutation operators, each over 30 independent runs. Saves results to `results/csv/ga_grid_search.csv`.

```bash
python -m main.grid_search
```

> Expected runtime: **~60–120 minutes**

---

### Step 3 — Inspect Grid Search Results

Generates plots from the grid search CSV so you can identify the best-performing operator configuration before running the final comparison.

```bash
python -m main.grid_search_visualizations
```

Produces:

- `results/visualizations/grid_search_top10.png` — top 10 configurations by F1-score
- `results/visualizations/grid_search_operators_avg.png` — average F1 per selection / crossover / mutation method

---

### Step 4 — GA vs GWO Comparison

Runs both algorithms 30 times each using the best configuration identified in Step 3, prints a summary table with mean ± std for F1-Score, Accuracy, Precision and Recall, and performs an **Independent T-Test statistical significance test** on the F1 distributions. Saves per-run results to `results/csv/ga_vs_gwo.csv`.

```bash
python -m main.compare
```

> Expected runtime: **~5 minutes**

---

### Step 4 — Comparison Visualizations

Generates the GA vs GWO comparison plots from the CSV produced in Step 3.

```bash
python -m main.compare_visualizations
```

Produces:

- `results/visualizations/ga_vs_gwo_boxplot.png` — F1-score distribution across 30 runs
- `results/visualizations/ga_vs_gwo_metrics.png` — mean Accuracy, Precision, Recall, F1 side by side
