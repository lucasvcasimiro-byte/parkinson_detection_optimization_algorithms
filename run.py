import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

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
    parkinson = pd.read_csv('data/parkinsons_preprocessed.csv')

    X = parkinson.drop('status', axis=1).values
    y = parkinson['status'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

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
            print(f"{metric_name}: {metric_value}")

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
    parkinson = pd.read_csv('data/parkinsons_preprocessed.csv')

    X = parkinson.drop('status', axis=1).values
    y = parkinson['status'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

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
            print(f"{metric_name}: {metric_value}")

    if return_metrics:
        return best_solution, best_fitness, history, test_metrics

    return best_solution, best_fitness, history


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
    Runs the GA with every combination of initialization, crossover, and mutation.
    """
    initialization_methods = ['uniform', 'normal']
    crossover_operators = [
        ('arithmetic_crossover', arithmetic_crossover),
        ('simulated_binary_crossover', simulated_binary_crossover),
    ]
    mutation_operators = [
        ('gaussian_mutation', gaussian_mutation),
        ('uniform_continuous_mutation', uniform_continuous_mutation),
    ]

    results = []
    convergence_rows = []

    for init_method in initialization_methods:
        for crossover_name, crossover_func in crossover_operators:
            for mutation_name, mutation_func in mutation_operators:
                print(
                    f'\nRunning GA with init={init_method}, '
                    f'crossover={crossover_name}, mutation={mutation_name}'
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

                print(f"Best fitness: {best_fitness:.4f}")

    results = sorted(results, key=lambda item: item['best_train_fitness'], reverse=True)

    if save_results:
        output_dir = Path(results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(output_dir / 'ga_operator_comparison.csv', index=False)
        pd.DataFrame(convergence_rows).to_csv(output_dir / 'ga_convergence.csv', index=False)

    print('\nSummary, ordered by best fitness:')
    for result in results:
        print(
            f"{result['best_train_fitness']:.4f} | "
            f"init={result['init_method']} | "
            f"crossover={result['crossover']} | "
            f"mutation={result['mutation']}"
        )

    return results


if __name__ == '__main__':
    run_all_operator_combinations()
