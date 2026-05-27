import numpy as np


def genetic_algorithm(generate_solution, fitness_function, n_dimensions, model, X, y,
                      
                    pop_size=50,
    generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    selection_func=None,
    crossover_func=None,
    mutation_func=None,
    init_method='uniform',
    **kwargs
):
    """
    Genetic Algorithm for continuous neural network weight optimization. Logs history of
    best fitness per generation.
    """
    # Creating and evaluating population and respective fitnesses
    population = np.array([generate_solution(n_dimensions, initialization_method=init_method)for _ in range(pop_size)])
    fitnesses = np.array([fitness_function(individual, model, X, y) for individual in population])
    
    # Evolutionary tracking variables
    best_solution = None
    best_fitness = -float('inf')
    history = []


    # Main Generational Loop (Matches professor's pattern)
    for generation in range(generations):
        
        # 2. Evaluate fitness for the current population
        fitnesses = np.array([fitness_function(individual, model, X, y) for individual in population])
        
        # Keep track of the absolute global best solution
        for i in range(pop_size):
            if fitnesses[i] > best_fitness:
                best_fitness = fitnesses[i]
                best_solution = population[i].copy()  # Prevents memory cross-contamination
                
        history.append(best_fitness)
        print(f" Generation {generation + 1}/{generations} | Best F1-Score: {best_fitness:.4f}")

        # 3. Breed the next generation
        new_population = []
        
        # Elitism: Directly protect and preserve the best configuration found so far
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
        
    print(f"\n✓ GA completed. Best Fitness: {best_fitness:.4f}")
    return best_solution, best_fitness, history
