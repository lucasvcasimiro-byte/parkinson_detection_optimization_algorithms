import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from optimization_models.ga import genetic_algorithm
from optimization_models.ga_operators import (
    tournament_selection, arithmetic_crossover, uniform_continuous_mutation
)
from optimization_models.grey_wolf_optimizer import grey_wolf_optimizer
from main.utils import evaluate_solution, fitness_function, generate_solution, get_network_architecture

df = pd.read_csv('data/parkinsons_preprocessed.csv')
X = df.drop('status', axis=1).values
y = df['status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build network with the best architecture found in the grid search
model, n_dimensions = get_network_architecture((10,), X_train, y_train)

# GA vs GWO comparison
ga_f1s, ga_accs, ga_precs, ga_recs = [], [], [], []
gwo_f1s, gwo_accs, gwo_precs, gwo_recs = [], [], [], []

for _ in range(30):

    # GA run — using the best operators from the grid search
    best_ga, _, _ = genetic_algorithm(
        generate_solution=generate_solution,
        fitness_function=fitness_function,
        n_dimensions=n_dimensions,
        model=model,
        X=X_train,
        y=y_train,
        pop_size=50,
        generations=100,

        # Best combination from the grid search, and also the better performing operators on average
        selection_func=tournament_selection,           
        crossover_func=arithmetic_crossover,      
        mutation_func=uniform_continuous_mutation, 
        init_method='normal',
        verbose=False,
    )

    # GWO run
    gwo, _, _ = grey_wolf_optimizer(
        generate_solution=generate_solution,
        fitness_function=fitness_function,
        n_dimensions=n_dimensions,
        model=model,
        X=X_train,
        y=y_train,
        pop_size=50,
        generations=100,
        verbose=False,
    )

    ga_metrics  = evaluate_solution(best_ga,  model, X_test, y_test)
    gwo_metrics = evaluate_solution(gwo, model, X_test, y_test)

    ga_f1s.append(ga_metrics['f1_score'])   
    ga_accs.append(ga_metrics['accuracy'])
    ga_precs.append(ga_metrics['precision']) 
    ga_recs.append(ga_metrics['recall'])

    gwo_f1s.append(gwo_metrics['f1_score'])   
    gwo_accs.append(gwo_metrics['accuracy'])
    gwo_precs.append(gwo_metrics['precision']) 
    gwo_recs.append(gwo_metrics['recall'])


# Summary table
print('Summary (30 runs)')
print(f"{'Metric':<15} {'GA Mean':>10} {'GA Std':>10} {'GWO Mean':>10} {'GWO Std':>10}")
print('-' * 60)
for metric, ga_vals, gwo_vals in [
    ('F1-Score',  ga_f1s,   gwo_f1s),
    ('Accuracy',  ga_accs,  gwo_accs),
    ('Precision', ga_precs, gwo_precs),
    ('Recall',    ga_recs,  gwo_recs),
]:
    print(f"{metric:<15} {np.mean(ga_vals):>10.4f} {np.std(ga_vals):>10.4f} {np.mean(gwo_vals):>10.4f} {np.std(gwo_vals):>10.4f}")

# Save results to CSV
results = []
for run in range(30):
    results.append({'run': run + 1, 'algorithm': 'GA',  'f1_score': ga_f1s[run],  'accuracy': ga_accs[run],  'precision': ga_precs[run],  'recall': ga_recs[run]})
    results.append({'run': run + 1, 'algorithm': 'GWO', 'f1_score': gwo_f1s[run], 'accuracy': gwo_accs[run], 'precision': gwo_precs[run], 'recall': gwo_recs[run]})

pd.DataFrame(results).to_csv('results/csv/ga_vs_gwo.csv', index=False)
