import pandas as pd

from optimization_models.ga import genetic_algorithm
from optimization_models.ga_operators import (
    arithmetic_crossover,
    gaussian_mutation,
    simulated_binary_crossover,
    tournament_selection,
    uniform_continuous_mutation,
)
from utils import fitness_function, generate_solution, get_network_architecture


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
        verbose=True):
    """
    Runs the Genetic Algorithm to optimize the weights of an MLPClassifier.
    """
    parkinson = pd.read_csv('data/parkinsons_preprocessed.csv')

    X = parkinson.drop('status', axis=1).values
    y = parkinson['status'].values

    model, n_dimensions = get_network_architecture(chosen_architecture, X, y)

    best_solution, best_fitness, history = genetic_algorithm(
        generate_solution=generate_solution,
        fitness_function=fitness_function,
        n_dimensions=n_dimensions,
        model=model,
        X=X,
        y=y,
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

    return best_solution, best_fitness, history


def run_all_operator_combinations():
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

    for init_method in initialization_methods:
        for crossover_name, crossover_func in crossover_operators:
            for mutation_name, mutation_func in mutation_operators:
                print(
                    f'\nRunning GA with init={init_method}, '
                    f'crossover={crossover_name}, mutation={mutation_name}'
                )

                best_solution, best_fitness, history = run_ga_experiment(
                    chosen_architecture=(10,),
                    pop_size=50,
                    generations=100,
                    crossover_rate=0.8,
                    mutation_rate=0.1,
                    init_method=init_method,
                    crossover_func=crossover_func,
                    mutation_func=mutation_func,
                    low=-1,
                    high=1,
                    tournament_size=3,
                    sigma=0.1,
                    verbose=False,
                )

                result = {
                    'init_method': init_method,
                    'crossover': crossover_name,
                    'mutation': mutation_name,
                    'best_fitness': best_fitness,
                    'initial_best_fitness': history[0],
                    'final_best_fitness': history[-1],
                    'solution_dimensions': len(best_solution),
                }
                results.append(result)

                print(f"Best fitness: {best_fitness:.4f}")

    results = sorted(results, key=lambda item: item['best_fitness'], reverse=True)

    print('\nSummary, ordered by best fitness:')
    for result in results:
        print(
            f"{result['best_fitness']:.4f} | "
            f"init={result['init_method']} | "
            f"crossover={result['crossover']} | "
            f"mutation={result['mutation']}"
        )

    return results


if __name__ == '__main__':
    run_all_operator_combinations()
