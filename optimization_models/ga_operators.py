import numpy as np

### SELECTION OPERATOR

def tournament_selection(population, fitnesses, tournament_size=3, **kwargs):
    """
    Selects an individual from the population using tournament selection.
    Used in class
    """
    pop_size = len(population)
    tournament_size = min(tournament_size, pop_size)
    # Randomly select tournament participants
    participants = np.random.choice(pop_size, tournament_size, replace=False)

    # Find best fitness among participants
    best_idx = participants[np.argmax(fitnesses[participants])]
    return population[best_idx]


def roulette_wheel_selection(population, fitnesses, **kwargs):
    """
    Selects an individual with a probability proportional to their fitness.
    Like spinning a roulette wheel where better solutions get bigger slices.
    """
    # Ensure all fitness values are positive for probability calculation
    min_fit = np.min(fitnesses)
    if min_fit < 0:
        adjusted_fitness = fitnesses - min_fit
    else:
        adjusted_fitness = fitnesses
        
    total_fitness = np.sum(adjusted_fitness)
    
    # Fallback if all fitnesses are identical or zero
    if total_fitness == 0:
        return population[np.random.randint(len(population))]
        
    probabilities = adjusted_fitness / total_fitness
    chosen_idx = np.random.choice(len(population), p=probabilities)
    
    return population[chosen_idx]


### CROSSOVER OPERATORS

def one_point_crossover(parent1, parent2, **kwargs):
    """
    Splits parents at a random point and creates two children by swapping the tails.
    Used in class
    """
    # Randomly pick a crossover point
    point = np.random.randint(1, len(parent1) - 1)

    # Swap the tails to create offspring
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def arithmetic_crossover(parent1, parent2, **kwargs):
    """
    Creates two children by taking a weighted average of the parents' genes, with 
    a random alpha parameter defining the ratio
    """
    # Generate a random alpha for blending the parents
    alpha = np.random.rand()

    # Create children as weighted averages of the parents
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2


def simulated_binary_crossover(parent1, parent2, eta=2, **kwargs):
    """
    Acts similiar to one point crossover, but treats each gene independently
    drawing a spread factor to blend the genes so that offspring is mostly created close to the 
    parents, but can occasionally expand further outwards to explore new regions
    """
    # Create empty arrays for the children
    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent2)

    # Generate random numbers for each parameter to determine the spread factor (beta)
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
    Mutates a solution by adding random gaussian noise to its genes to help refine and 
    locally improve existing solutions.
    More exploitative
    """
    # Create a copy so we don't modify the original parent directly
    mutated_solution = np.copy(solution)
    
    # Iterate through each weight (gene) in the solution
    for i in range(len(mutated_solution)):
        if np.random.rand() < mutation_rate:
            # Add random gaussian noise
            mutated_solution[i] += np.random.normal(0, sigma)
            
    # boundary check -------   acho q deviamos apagar estes
    # Force any exploding weights back inside the [low, high] limits
    # mutated_solution = np.clip(mutated_solution, low, high)
    
    return mutated_solution


def uniform_continuous_mutation(solution, mutation_rate, low=-1, high=1, **kwargs):
    """
    Mutates a solution by replacing its value with a new random 
    number within the boundaries.
    More explorative
    """
    # Create a copy so we don't modify the original parent directly
    mutated_solution = np.copy(solution)
    
    # Iterate through each weight (gene) in the solution
    for i in range(len(mutated_solution)):
        if np.random.rand() < mutation_rate:
            # Replace the weight with a completely new random value within bounds
            mutated_solution[i] = np.random.uniform(low, high)
    
    return mutated_solution
