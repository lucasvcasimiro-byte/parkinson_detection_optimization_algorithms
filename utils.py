import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

def get_network_architecture(chosen_architecture, X, y):
    """
    Given a chosen architecture, initializes a MLPClassifier with the appropriate number
    of layers and neurons per layer, and calculates the total number of weights (dimensions).
    
    Args:
        chosen_architecture: Tuple of integers representing hidden layer sizes.
                            Example: (10,) for 1 hidden layer with 10 neurons
                                     (10, 5) for 2 hidden layers with 10 and 5 neurons
        X: Input feature matrix of shape (n_samples, n_features)
        y: Target labels
    
    Returns:
        model: Initialized MLPClassifier with the specified architecture
        n_dimensions: Total number of weights and biases in the network
    """
    # Determine network dimensions
    n_input = X.shape[1]
    n_output = len(np.unique(y))
    
    # Create the full layer architecture: input -> hidden layers -> output
    layer_sizes = [n_input] + list(chosen_architecture) + [n_output]
    
    # Initialize MLPClassifier with the hidden layer sizes (not including input/output)
    model = MLPClassifier(
        hidden_layer_sizes=chosen_architecture,
        max_iter=1000,
        random_state=42,
        warm_start=False
    )
    
    # Fit the model once to initialize weights (required for weight extraction)
    model.fit(X, y)
    
    # Calculate total number of weights and biases
    # For each layer transition: weights + biases
    n_dimensions = 0
    for i in range(len(layer_sizes) - 1):
        # Weights between layer i and layer i+1
        n_dimensions += layer_sizes[i] * layer_sizes[i + 1]
        # Biases for layer i+1
        n_dimensions += layer_sizes[i + 1]
    
    return model, n_dimensions

def generate_solution(n_dimensions, initialization_method = 'uniform', low = -1, high = 1):
    """
    Produces a random weight vector of the appropriate length for the
    chosen network architecture.
    """
    # 1st initialization method: uniform
    if initialization_method == 'uniform':
        return np.random.uniform(low, high, n_dimensions)
    #2nd initialization method: normal
    elif initialization_method == 'normal':
        return np.random.normal(0,1, n_dimensions)
    else:
        raise ValueError("Invalid initialization method. Choose 'uniform' or 'normal'.")
    
def vector_to_weights(vector, mlp):
    """
    Converts a flat weight vector into the weights and biases format used by MLPClassifier.
    
    Args:
        vector: Flat array of all weights and biases concatenated
        mlp: Fitted MLPClassifier object (used to know the expected shapes)
    
    Returns:
        coefs: List of weight matrices for each layer
        intercepts: List of bias vectors for each layer
    """
    coefs = []
    intercepts = []
    
    idx = 0
    # Process each layer transition
    for i in range(len(mlp.coefs_)):
        # Get the expected shape of weights for this layer
        shape = mlp.coefs_[i].shape
        n_weights = shape[0] * shape[1]
        
        # Extract weights and reshape
        weights = vector[idx:idx + n_weights].reshape(shape)
        coefs.append(weights)
        idx += n_weights
        
        # Get the expected shape of biases for this layer
        bias_shape = mlp.intercepts_[i].shape[0]
        
        # Extract biases
        biases = vector[idx:idx + bias_shape]
        intercepts.append(biases)
        idx += bias_shape
    
    return coefs, intercepts


def fitness_function(vector, mlp, X, y):
    """
    Given a weight vector, loads the weights into the network, runs
    predictions on the dataset, and returns a value reflecting the predictive performance of the configuration
    """
    # Convert flat vector to weights and biases
    coefs, intercepts = vector_to_weights(vector, mlp)
    mlp.coefs_ = coefs
    mlp.intercepts_ = intercepts
    pred = mlp.predict(X)

    # Temos de ver qual a melhor metrica para por no average(depende do dataset)
    score = f1_score(y,pred, average='weighted')
    return score