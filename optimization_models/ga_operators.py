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

def gaussian_mutation(individual, mutation_rate=0.01, sigma=0.1, low=None, high=None, **kwargs):
    """
    Adds small random noise sampled from a Gaussian distribution 
    to selected genes to perform local optimization/tuning.
    """
    mutated = np.copy(individual)
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i] += np.random.normal(0, sigma)
    if low is not None or high is not None:
        mutated = np.clip(mutated, low if low is not None else -np.inf, high if high is not None else np.inf)
    return mutated

def uniform_continuous_mutation(individual, mutation_rate=0.01, low=-1, high=1, **kwargs):
    """
    Completely replaces selected genes with new random values drawn from a uniform scale.
    """
    mutated = np.copy(individual)
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i] = np.random.uniform(low, high)
    return mutated
