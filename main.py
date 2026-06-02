import pandas as pd
from sklearn.model_selection import train_test_split

from optimization_models.ga import genetic_algorithm
from optimization_models.ga_operators import arithmetic_crossover, gaussian_mutation, tournament_selection
from optimization_models.grey_wolf_optimizer import grey_wolf_optimizer
from utils import evaluate_solution, fitness_function, generate_solution, get_network_architecture

# Load data
df = pd.read_csv('data/parkinsons_preprocessed.csv')
X = df.drop('status', axis=1).values
y = df['status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build network
model, n_dimensions = get_network_architecture((10,), X_train, y_train)

# Run GA
best_solution, best_fitness, history = genetic_algorithm(
    generate_solution=generate_solution,
    fitness_function=fitness_function,
    n_dimensions=n_dimensions,
    model=model,
    X=X_train,
    y=y_train,
    pop_size=50,
    generations=100,
    selection_func=tournament_selection,
    crossover_func=arithmetic_crossover,
    mutation_func=gaussian_mutation,
)

# Evaluate on test set
metrics = evaluate_solution(best_solution, model, X_test, y_test)
print("GA Test Results:", metrics)

# Run GWO
best_solution_gwo, best_fitness_gwo, history_gwo = grey_wolf_optimizer(
    generate_solution=generate_solution,
    fitness_function=fitness_function,
    n_dimensions=n_dimensions,
    model=model,
    X=X_train,
    y=y_train,
    pop_size=50,
    generations=100,
)

metrics_gwo = evaluate_solution(best_solution_gwo, model, X_test, y_test)
print("GWO Test Results:", metrics_gwo)
