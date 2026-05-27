import pandas as pd
from utils import get_network_architecture, fitness_function, generate_solution

parkinson = pd.read_csv('data/parkinsons_preprocessed.csv')

X = parkinson.drop('status', axis=1).values
y = parkinson['status'].values

#Exemplo
chosen_architecture = (10,)

model, n_dimensions = get_network_architecture(chosen_architecture, X, y)

random_solution = generate_solution(n_dimensions, initialization_method='uniform', low=-1, high=1)

fitness = fitness_function(random_solution, model, X, y)