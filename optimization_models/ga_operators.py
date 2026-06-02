import numpy as np

### SELECTION OPERATOR

def tournament_selection(population, fitnesses, tournament_size=3, **kwargs):
    """
    Selects an individual from the population using tournament selection.
    """
    pop_size = len(population)
    tournament_size = min(tournament_size, pop_size)
    # Randomly select tournament participants
    participants = np.random.choice(pop_size, tournament_size, replace=False)

    # Find best fitness among participants
    best_idx = participants[np.argmax(fitnesses[participants])]
    return population[best_idx]


### CROSSOVER OPERATORS

def arithmetic_crossover(parent1, parent2, **kwargs):
    """
    Produces two offspring that fall along the line connecting the parents.
    """
    alpha = np.random.rand()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2

def one_point_crossover(parent1, parent2, **kwargs):

    # Randomly pick a crossover point
    xover_point = np.random.randint(1, len(parent1) - 1)

    # when parent1 and parent 2 love each other very much...
    # note: np.concatenate instead of + because these are numpy arrays, not lists
    child1 = np.concatenate([parent1[:xover_point], parent2[xover_point:]])
    child2 = np.concatenate([parent2[:xover_point], parent1[xover_point:]])

    # returning the offspring
    return child1, child2

def simulated_binary_crossover(parent1, parent2, eta=2, **kwargs):
    """
    Performs simulated binary crossover (SBX) between two parents.
    """
    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent2)

    # Generate random numbers for each parameter to determine the spread factor (beta
    u = np.random.random(size=len(parent1))

    #Calculate beta based on distribution index eta
    beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)), (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))

    # Create the offspring using the spread factor
    child1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
    child2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)
    return child1, child2


### MUTATION OPERATORS

def gaussian_mutation(solution, mutation_rate, low=-1, high=1, sigma=0.1, **kwargs):
    """
    Mutates a solution by adding Gaussian noise to its genes.
    """
    # Create a copy so we don't modify the original parent directly
    mutated_solution = np.copy(solution)
    
    # Iterate through each weight (gene) in the solution
    for i in range(len(mutated_solution)):
        if np.random.rand() < mutation_rate:
            # Add random Gaussian noise
            mutated_solution[i] += np.random.normal(0, sigma)
            
    # --- BOUNDARY CHECK ---
    # Force any exploding weights back inside the [low, high] limits
    mutated_solution = np.clip(mutated_solution, low, high)
    
    return mutated_solution

def uniform_continuous_mutation(solution, mutation_rate, low=-1, high=1, **kwargs):
    """
    Mutates a solution by replacing genes with a random uniform value.
    """
    mutated_solution = np.copy(solution)
    
    for i in range(len(mutated_solution)):
        if np.random.rand() < mutation_rate:
            # Replace the weight with a completely new random value within bounds
            mutated_solution[i] = np.random.uniform(low, high)
            
    # --- BOUNDARY CHECK (Failsafe) ---
    mutated_solution = np.clip(mutated_solution, low, high)
    
    return mutated_solution
