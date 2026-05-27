import pandas as pd
import algorithms


parkinson = pd.read_csv('data/parkinsons_preprocessed.csv')

X = parkinson.drop('status', axis=1).values
y = parkinson['status'].values

chosen_architecture = (10,)

model, n_dimensions = algorithms.get_network_architecture(chosen_architecture, X, y)