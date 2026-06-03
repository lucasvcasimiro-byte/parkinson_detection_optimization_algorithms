import numpy as np


def grey_wolf_optimizer(
    generate_solution,
    fitness_function,
    n_dimensions,
    model,
    X,
    y,
    pop_size=50,
    generations=100,
    init_method="uniform",
    low=-1,
    high=1,
    verbose=True,
    **kwargs
):
    """
    Grey Wolf Optimizer for continuous neural-network weight optimization.

    Each wolf is a real-valued vector representing all MLP weights and biases, where the best three wolves 
    guide the search (replicating a wolf pack's hunting behavior): alpha is the best solution, beta is the second best, 
    and delta is the third. The rest follow the leader. Also logs history of best fitness per generation
    """
    # At least 3 wolves are required to function properly (so this is only a safety check for reproducibility)
    if pop_size < 3:
        raise ValueError("Grey Wolf Optimizer requires population bigger than 3.")


    # Initialize population 
    population = np.array([
        generate_solution(
            n_dimensions,
            initialization_method=init_method,
            low=low,
            high=high
        )
        for _ in range(pop_size)
    ])
    population = np.clip(population, low, high)
    fitnesses = np.array([
        fitness_function(wolf, model, X, y)
        for wolf in population
    ])

    # Sort wolves by fitness to identify alpha, beta and delta
    sorted_indices = np.argsort(fitnesses)[::-1]
    alpha_position = population[sorted_indices[0]].copy()
    beta_position = population[sorted_indices[1]].copy()
    delta_position = population[sorted_indices[2]].copy()
    alpha_fitness = fitnesses[sorted_indices[0]]
    beta_fitness = fitnesses[sorted_indices[1]]
    delta_fitness = fitnesses[sorted_indices[2]]

    # Store history
    history = []

    # Main optimization loop
    for generation in range(generations):
        if generations > 1:
            a = 2 - (2 * generation / (generations - 1))
        else:
            a = 2

        new_population = []

        # Update each wolf's position based on leaders influence
        for wolf in population:
            # Alpha influence
            r1 = np.random.random(n_dimensions)
            r2 = np.random.random(n_dimensions)
            a1 = 2 * a * r1 - a
            c1 = 2 * r2
            d_alpha = np.abs(c1 * alpha_position - wolf)
            x1 = alpha_position - a1 * d_alpha

            # Beta influence, 
            r1 = np.random.random(n_dimensions)
            r2 = np.random.random(n_dimensions)
            a2 = 2 * a * r1 - a
            c2 = 2 * r2
            d_beta = np.abs(c2 * beta_position - wolf)
            x2 = beta_position - a2 * d_beta

            # Delta influence
            r1 = np.random.random(n_dimensions)
            r2 = np.random.random(n_dimensions)
            a3 = 2 * a * r1 - a
            c3 = 2 * r2
            d_delta = np.abs(c3 * delta_position - wolf)
            x3 = delta_position - a3 * d_delta

            new_position = (x1 + x2 + x3) / 3

            # Clip weights back to boundaries in case of explosion
            new_population.append(np.clip(new_position, low, high))

        # Transition to the new generated population
        population = np.array(new_population)
        fitnesses = np.array([
            fitness_function(wolf, model, X, y)
            for wolf in population
        ])

        # Update alpha, beta and delta based on the new fitnesses
        for wolf, fitness in zip(population, fitnesses):
            if fitness > alpha_fitness:
                delta_position = beta_position.copy()
                delta_fitness = beta_fitness
                beta_position = alpha_position.copy()
                beta_fitness = alpha_fitness
                alpha_position = wolf.copy()
                alpha_fitness = fitness
            elif fitness > beta_fitness:
                delta_position = beta_position.copy()
                delta_fitness = beta_fitness
                beta_position = wolf.copy()
                beta_fitness = fitness
            elif fitness > delta_fitness:
                delta_position = wolf.copy()
                delta_fitness = fitness

        history.append(alpha_fitness)

    if verbose:
        print(f"Best Fitness: {alpha_fitness:.4f}")
    return alpha_position, alpha_fitness, history
