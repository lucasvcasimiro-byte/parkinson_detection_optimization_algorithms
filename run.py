import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from optimization_models.ga import genetic_algorithm
from optimization_models.ga_operators import (
    arithmetic_crossover,
    gaussian_mutation,
    simulated_binary_crossover,
    tournament_selection,
    uniform_continuous_mutation,
)
from optimization_models.grey_wolf_optimizer import grey_wolf_optimizer
from utils import evaluate_solution, fitness_function, generate_solution, get_network_architecture


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_data(test_size=0.2, random_state=42):
    """Loads and splits the Parkinson's dataset."""
    parkinson = pd.read_csv('data/parkinsons_preprocessed.csv')
    X = parkinson.drop('status', axis=1).values
    y = parkinson['status'].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ---------------------------------------------------------------------------
# Single-run experiment helpers
# ---------------------------------------------------------------------------

def run_ga_experiment(
        chosen_architecture=(10,),
        pop_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        init_method='uniform',
        crossover_func=arithmetic_crossover,
        mutation_func=gaussian_mutation,
        low=-1,
        high=1,
        tournament_size=3,
        sigma=0.1,
        verbose=True,
        test_size=0.2,
        random_state=42,
        return_metrics=False):
    """
    Runs the Genetic Algorithm to optimize the weights of an MLPClassifier.
    """
    X_train, X_test, y_train, y_test = _load_data(test_size, random_state)
    model, n_dimensions = get_network_architecture(chosen_architecture, X_train, y_train)

    best_solution, best_fitness, history = genetic_algorithm(
        generate_solution=generate_solution,
        fitness_function=fitness_function,
        n_dimensions=n_dimensions,
        model=model,
        X=X_train,
        y=y_train,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        selection_func=tournament_selection,
        crossover_func=crossover_func,
        mutation_func=mutation_func,
        init_method=init_method,
        low=low,
        high=high,
        tournament_size=tournament_size,
        sigma=sigma,
        verbose=verbose,
    )

    test_metrics = evaluate_solution(best_solution, model, X_test, y_test)

    if verbose:
        print("Test metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {metric_value}")

    if return_metrics:
        return best_solution, best_fitness, history, test_metrics

    return best_solution, best_fitness, history


def run_gwo_experiment(
        chosen_architecture=(10,),
        pop_size=50,
        generations=100,
        init_method="uniform",
        low=-1,
        high=1,
        test_size=0.2,
        random_state=42,
        verbose=True,
        return_metrics=False):
    """
    Runs the Grey Wolf Optimizer to optimize the weights of an MLPClassifier.
    """
    X_train, X_test, y_train, y_test = _load_data(test_size, random_state)
    model, n_dimensions = get_network_architecture(chosen_architecture, X_train, y_train)

    best_solution, best_fitness, history = grey_wolf_optimizer(
        generate_solution=generate_solution,
        fitness_function=fitness_function,
        n_dimensions=n_dimensions,
        model=model,
        X=X_train,
        y=y_train,
        pop_size=pop_size,
        generations=generations,
        init_method=init_method,
        low=low,
        high=high,
        verbose=verbose,
    )

    test_metrics = evaluate_solution(best_solution, model, X_test, y_test)

    if verbose:
        print("Test metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {metric_value}")

    if return_metrics:
        return best_solution, best_fitness, history, test_metrics

    return best_solution, best_fitness, history


# ---------------------------------------------------------------------------
# Grid-comparison experiments
# ---------------------------------------------------------------------------

def run_all_operator_combinations(
        chosen_architecture=(10,),
        pop_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        low=-1,
        high=1,
        tournament_size=3,
        sigma=0.1,
        test_size=0.2,
        random_state=42,
        results_dir="results",
        save_results=True):
    """
    Runs the GA with every combination of initialization method, crossover
    operator, and mutation operator (2 init × 2 crossover × 2 mutation = 8 runs).

    Crossover operators: arithmetic_crossover, simulated_binary_crossover
    Mutation operators:  gaussian_mutation, uniform_continuous_mutation
    """
    initialization_methods = ['uniform', 'normal']

    crossover_operators = [
        ('arithmetic_crossover',       arithmetic_crossover),
        ('simulated_binary_crossover', simulated_binary_crossover),
    ]

    mutation_operators = [
        ('gaussian_mutation',           gaussian_mutation),
        ('uniform_continuous_mutation', uniform_continuous_mutation),
    ]

    results = []
    convergence_rows = []

    for init_method in initialization_methods:
        for crossover_name, crossover_func in crossover_operators:
            for mutation_name, mutation_func in mutation_operators:
                print(
                    f'\nRunning GA | init={init_method} | '
                    f'crossover={crossover_name} | mutation={mutation_name}'
                )

                start_time = time.perf_counter()
                best_solution, best_fitness, history, test_metrics = run_ga_experiment(
                    chosen_architecture=chosen_architecture,
                    pop_size=pop_size,
                    generations=generations,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    init_method=init_method,
                    crossover_func=crossover_func,
                    mutation_func=mutation_func,
                    low=low,
                    high=high,
                    tournament_size=tournament_size,
                    sigma=sigma,
                    verbose=False,
                    test_size=test_size,
                    random_state=random_state,
                    return_metrics=True,
                )
                runtime_seconds = time.perf_counter() - start_time

                for generation, fitness in enumerate(history, start=1):
                    convergence_rows.append({
                        'algorithm': 'GA',
                        'init_method': init_method,
                        'crossover': crossover_name,
                        'mutation': mutation_name,
                        'generation': generation,
                        'best_fitness': fitness,
                    })

                result = {
                    'algorithm': 'GA',
                    'architecture': chosen_architecture,
                    'pop_size': pop_size,
                    'generations': generations,
                    'init_method': init_method,
                    'crossover': crossover_name,
                    'mutation': mutation_name,
                    'best_train_fitness': best_fitness,
                    'initial_best_fitness': history[0],
                    'final_best_fitness': history[-1],
                    'test_weighted_f1': test_metrics['weighted_f1'],
                    'test_macro_f1': test_metrics['macro_f1'],
                    'test_balanced_accuracy': test_metrics['balanced_accuracy'],
                    'test_confusion_matrix': test_metrics['confusion_matrix'],
                    'solution_dimensions': len(best_solution),
                    'runtime_seconds': runtime_seconds,
                }
                results.append(result)

                print(f"  Best MCC: {best_fitness:.4f} | "
                      f"Weighted F1 (test): {test_metrics['weighted_f1']:.4f}")

    results = sorted(results, key=lambda item: item['best_train_fitness'], reverse=True)

    if save_results:
        output_dir = Path(results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(output_dir / 'ga_operator_comparison.csv', index=False)
        pd.DataFrame(convergence_rows).to_csv(output_dir / 'ga_convergence.csv', index=False)
        print(f"\nResults saved to {output_dir.resolve()}")

    print('\nSummary — GA operator comparison (sorted by best train fitness):')
    print(f"{'Init':<10} {'Crossover':<32} {'Mutation':<30} {'MCC':>8} {'W-F1':>8}")
    print('-' * 92)
    for r in results:
        print(
            f"{r['init_method']:<10} "
            f"{r['crossover']:<32} "
            f"{r['mutation']:<30} "
            f"{r['best_train_fitness']:>8.4f} "
            f"{r['test_weighted_f1']:>8.4f}"
        )

    return results


def run_gwo_comparison(
        chosen_architecture=(10,),
        pop_size=50,
        generations=100,
        low=-1,
        high=1,
        test_size=0.2,
        random_state=42,
        results_dir="results",
        save_results=True):
    """
    Runs the Grey Wolf Optimizer with both initialization methods and saves
    convergence history and summary metrics to CSV files for comparison.
    """
    initialization_methods = ['uniform', 'normal']

    results = []
    convergence_rows = []

    for init_method in initialization_methods:
        print(f'\nRunning GWO | init={init_method}')

        start_time = time.perf_counter()
        best_solution, best_fitness, history, test_metrics = run_gwo_experiment(
            chosen_architecture=chosen_architecture,
            pop_size=pop_size,
            generations=generations,
            init_method=init_method,
            low=low,
            high=high,
            test_size=test_size,
            random_state=random_state,
            verbose=False,
            return_metrics=True,
        )
        runtime_seconds = time.perf_counter() - start_time

        for generation, fitness in enumerate(history, start=1):
            convergence_rows.append({
                'algorithm': 'GWO',
                'init_method': init_method,
                'generation': generation,
                'best_fitness': fitness,
            })

        result = {
            'algorithm': 'GWO',
            'architecture': chosen_architecture,
            'pop_size': pop_size,
            'generations': generations,
            'init_method': init_method,
            'best_train_fitness': best_fitness,
            'initial_best_fitness': history[0],
            'final_best_fitness': history[-1],
            'test_weighted_f1': test_metrics['weighted_f1'],
            'test_macro_f1': test_metrics['macro_f1'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'],
            'test_confusion_matrix': test_metrics['confusion_matrix'],
            'solution_dimensions': len(best_solution),
            'runtime_seconds': runtime_seconds,
        }
        results.append(result)

        print(f"  Best MCC: {best_fitness:.4f} | "
              f"Weighted F1 (test): {test_metrics['weighted_f1']:.4f}")

    if save_results:
        output_dir = Path(results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(output_dir / 'gwo_results.csv', index=False)
        pd.DataFrame(convergence_rows).to_csv(output_dir / 'gwo_convergence.csv', index=False)
        print(f"\nGWO results saved to {output_dir.resolve()}")

    return results


# ---------------------------------------------------------------------------
# Statistical comparison (GA vs GWO) over multiple independent runs
# ---------------------------------------------------------------------------

def run_statistical_comparison(
        num_runs=10,
        results_dir="results",
        save_results=True):
    """
    Runs GA and GWO each `num_runs` times to collect a distribution of
    weighted F1 scores on the test set.

    Reports mean ± std and runs a two-sided Mann-Whitney U test to assess
    whether the difference between algorithms is statistically significant.
    The Mann-Whitney U test is non-parametric and makes no assumptions about
    the distribution of the scores, which is appropriate for a small sample.
    """
    print(f"\n--- Running {num_runs} independent executions of GA and GWO ---")

    ga_f1_results = []
    gwo_f1_results = []

    for i in range(num_runs):
        print(f"  Run {i + 1}/{num_runs}...")

        # GA — best operator configuration from comparison
        _, _, _, ga_metrics = run_ga_experiment(
            init_method='uniform',
            crossover_func=arithmetic_crossover,
            mutation_func=gaussian_mutation,
            verbose=False,
            return_metrics=True,
        )
        ga_f1_results.append(ga_metrics['weighted_f1'])

        # GWO
        _, _, _, gwo_metrics = run_gwo_experiment(
            init_method='uniform',
            verbose=False,
            return_metrics=True,
        )
        gwo_f1_results.append(gwo_metrics['weighted_f1'])

    # Descriptive statistics
    ga_mean, ga_std   = np.mean(ga_f1_results),  np.std(ga_f1_results)
    gwo_mean, gwo_std = np.mean(gwo_f1_results), np.std(gwo_f1_results)

    # Mann-Whitney U test (two-sided, non-parametric)
    u_stat, p_value = stats.mannwhitneyu(ga_f1_results, gwo_f1_results, alternative='two-sided')

    print("\n" + "=" * 56)
    print("  FINAL RESULTS — Weighted F1-Score over Test Set")
    print("=" * 56)
    print(f"  Genetic Algorithm:    {ga_mean:.4f} ± {ga_std:.4f}")
    print(f"  Grey Wolf Optimizer:  {gwo_mean:.4f} ± {gwo_std:.4f}")
    print("-" * 56)
    print(f"  Mann-Whitney U = {u_stat:.1f},  p-value = {p_value:.4f}")
    if p_value < 0.05:
        better = "GA" if ga_mean > gwo_mean else "GWO"
        print(f"  -> Statistically significant difference (p < 0.05).")
        print(f"     {better} performs significantly better.")
    else:
        print(f"  -> No statistically significant difference (p >= 0.05).")
        if gwo_mean > ga_mean:
            print("     GWO achieves a higher mean score, but the difference")
            print("     is not statistically significant at α = 0.05.")
        else:
            print("     GA achieves a higher mean score, but the difference")
            print("     is not statistically significant at α = 0.05.")
    print("=" * 56)

    if save_results:
        output_dir = Path(results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            'run': list(range(1, num_runs + 1)),
            'ga_weighted_f1': ga_f1_results,
            'gwo_weighted_f1': gwo_f1_results,
        })
        df.to_csv(output_dir / 'statistical_comparison.csv', index=False)

        summary = pd.DataFrame([{
            'algorithm': 'GA',
            'mean_weighted_f1': ga_mean,
            'std_weighted_f1': ga_std,
            'num_runs': num_runs,
            'mann_whitney_u': u_stat,
            'p_value': p_value,
        }, {
            'algorithm': 'GWO',
            'mean_weighted_f1': gwo_mean,
            'std_weighted_f1': gwo_std,
            'num_runs': num_runs,
            'mann_whitney_u': u_stat,
            'p_value': p_value,
        }])
        summary.to_csv(output_dir / 'statistical_summary.csv', index=False)
        print(f"\nStatistical results saved to {output_dir.resolve()}")

    return ga_f1_results, gwo_f1_results


# ---------------------------------------------------------------------------
# Entry point — full reproducible run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("  STEP 1 — GA Operator Comparison")
    print("=" * 60)
    run_all_operator_combinations()

    print("\n" + "=" * 60)
    print("  STEP 2 — GWO Comparison (init methods)")
    print("=" * 60)
    run_gwo_comparison()

    print("\n" + "=" * 60)
    print("  STEP 3 — Statistical Comparison (GA vs GWO, 10 runs)")
    print("=" * 60)
    run_statistical_comparison(num_runs=10)
