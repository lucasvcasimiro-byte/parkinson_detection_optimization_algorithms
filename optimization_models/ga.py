import numpy as np
from optimization_models.ga_operators import (arithmetic_crossover, gaussian_mutation, tournament_selection)


def genetic_algorithm(generate_solution, fitness_function, n_dimensions, model, X, y,
                      pop_size=50, 
                      generations=100,
                      crossover_rate=0.8, 
                      mutation_rate=0.1,
                      selection_func = None,
                      crossover_func = None,
                      mutation_func = None,
                      init_method = 'uniform',
                      verbose=True,
                      **kwargs):
    """
    Genetic Algorithm for continuous neural network weight optimization. Logs history of
    best fitness per generation
    """
    if selection_func is None:
        selection_func = tournament_selection
    if crossover_func is None:
        crossover_func = arithmetic_crossover
    if mutation_func is None:
        mutation_func = gaussian_mutation

    low = kwargs.get('low', -1)
    high = kwargs.get('high', 1)

    # Creating and evaluating population and respective fitnesses
    population = np.array([generate_solution(n_dimensions, initialization_method=init_method, low=low, high=high) for _ in range(pop_size)])
    fitnesses = np.array([fitness_function(individual, model, X, y) for individual in population])
    
    # Tracking variables
    best_solution = None
    best_fitness = -float('inf')
    history = []

    # Main evolution loop
    for generation in range(generations):     
        for i in range(pop_size):
            fitnesses[i] = fitness_function(population[i], model, X, y)
            # Keep track of the absolute global best solution
            if fitnesses[i] > best_fitness:
                best_fitness = fitnesses[i]
                best_solution = population[i].copy()  
                
        history.append(best_fitness)

        # Create next generation
        new_population = []
        

        # Elitism, passing best results directly into the next generation
        new_population.append(best_solution.copy())
        
        while len(new_population) < pop_size:
            # Selection Phase
            parent1 = selection_func(population, fitnesses, **kwargs)
            parent2 = selection_func(population, fitnesses, **kwargs)
            
            # Crossover Phase
            if np.random.random() < crossover_rate:
                child1, child2 = crossover_func(parent1, parent2, **kwargs)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
                
            # Mutation Phase
            child1 = mutation_func(child1, mutation_rate=mutation_rate, **kwargs)
            child2 = mutation_func(child2, mutation_rate=mutation_rate, **kwargs)
            
            # Append offspring to the next generation pool
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
                
        # Transition to the newly generated population
        population = np.array(new_population)
        
    if verbose:
        print(f"Best Fitness: {best_fitness:.4f}")
    return best_solution, best_fitness, history
