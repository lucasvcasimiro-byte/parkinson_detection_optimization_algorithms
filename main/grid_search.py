import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from optimization_models.ga import genetic_algorithm
from optimization_models.ga_operators import (
    arithmetic_crossover, one_point_crossover, simulated_binary_crossover,
    gaussian_mutation, uniform_continuous_mutation,
    tournament_selection, roulette_wheel_selection
)
from optimization_models.grey_wolf_optimizer import grey_wolf_optimizer
from main.utils import evaluate_solution, fitness_function, generate_solution, get_network_architecture

df = pd.read_csv('data/parkinsons_preprocessed.csv')
X = df.drop('status', axis=1).values
y = df['status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

"""
# Don't run this, it takes some time, 30 iterations for statistical significance



# Grid search over operators and initialization methods
architectures = [(10,), (20,), (10, 5)] # 1-layer and 2-layer examples
selections   = [tournament_selection, roulette_wheel_selection]
crossovers   = [arithmetic_crossover, one_point_crossover, simulated_binary_crossover]
mutations    = [gaussian_mutation, uniform_continuous_mutation]
initializations = ['uniform', 'normal']

results = []

# Table in terminal to know where the process is while running
print(f"{'Arch':<15} {'Init':<10} {'Selection':<25} {'Crossover':<35} {'Mutation':<30} {'Avg F1':>10} {'Std F1':>10}")
print('-' * 145)

for arch in architectures:
    # Build network for this specific architecture
    model, n_dimensions = get_network_architecture(arch, X_train, y_train)
    
    for init_method in initializations:
        for selection_func in selections:
            for crossover_func in crossovers:
                for mutation_func in mutations:
                    
                    run_f1s = []
                    run_accs = []
                    run_precs = []
                    run_recs = []
                
                    for run_idx in range(30):
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
                        run_f1s.append(metrics['f1_score'])
                        run_accs.append(metrics['accuracy'])
                        run_precs.append(metrics['precision'])
                        run_recs.append(metrics['recall'])

                    avg_f1 = np.mean(run_f1s)
                    std_f1 = np.std(run_f1s)
                    
                    results.append({
                        'architecture':      str(arch),
                        'init_method':       init_method,
                        'selection':         selection_func.__name__,
                        'crossover':         crossover_func.__name__,
                        'mutation':          mutation_func.__name__,
                        'accuracy':          np.mean(run_accs),
                        'precision':         np.mean(run_precs),
                        'recall':            np.mean(run_recs),
                        'f1_score':          avg_f1,
                        'f1_std':            std_f1
                    })
                    
                    print(f"{str(arch):<15} {init_method:<10} {selection_func.__name__:<25} {crossover_func.__name__:<35} {mutation_func.__name__:<30} {avg_f1:>10.4f} {std_f1:>10.4f}")
                    

pd.DataFrame(results).to_csv('results/csv/ga_grid_search.csv', index=False)
"""
