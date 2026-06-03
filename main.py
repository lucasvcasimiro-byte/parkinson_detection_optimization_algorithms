import pandas as pd
from sklearn.model_selection import train_test_split

from optimization_models.ga import genetic_algorithm
from optimization_models.ga_operators import (
    arithmetic_crossover, one_point_crossover, simulated_binary_crossover,
    gaussian_mutation, uniform_continuous_mutation,
    tournament_selection, roulette_wheel_selection
)
from optimization_models.grey_wolf_optimizer import grey_wolf_optimizer
from utils import evaluate_solution, fitness_function, generate_solution, get_network_architecture

# Load data
df = pd.read_csv('data/parkinsons_preprocessed.csv')
X = df.drop('status', axis=1).values
y = df['status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build network
model, n_dimensions = get_network_architecture((10,), X_train, y_train)

# Grid search over operators and initialization methods
selections   = [tournament_selection, roulette_wheel_selection]
crossovers   = [arithmetic_crossover, one_point_crossover, simulated_binary_crossover]
mutations    = [gaussian_mutation, uniform_continuous_mutation]
initializations = ['uniform', 'normal']


results = []

for init_method in initializations:
    for selection_func in selections:
        for crossover_func in crossovers:
            for mutation_func in mutations:
                best_solution, _, _ = genetic_algorithm(
                    generate_solution=generate_solution,
                    fitness_function=fitness_function,
                    n_dimensions=n_dimensions,
                    model=model,
                    X=X_train,
                    y=y_train,
                    pop_size=50,
                    generations=100,
                    selection_func=selection_func,
                    crossover_func=crossover_func,
                    mutation_func=mutation_func,
                    init_method=init_method,
                    verbose=False,
                )
                metrics = evaluate_solution(best_solution, model, X_test, y_test)
                results.append({
                    'init_method':       init_method,
                    'selection':         selection_func.__name__,
                    'crossover':         crossover_func.__name__,
                    'mutation':          mutation_func.__name__,
                    'accuracy':          metrics['accuracy'],
                    'precision':         metrics['precision'],
                    'recall':            metrics['recall'],
                    'f1_score':          metrics['f1_score'],
                })
                

pd.DataFrame(results).to_csv('results/ga_grid_search.csv', index=False)
print("\nSaved: results/ga_operators_grid_search.csv")
