import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

def get_network_architecture(chosen_architecture, X, y):
    """
    Given a chosen architecture, initializes a MLPClassifier with the appropriate number
    of layers and neurons per layer, and calculates the total number of weights (dimensions)
    """
    # Initialize MLPClassifier with the hidden layer sizes 
    model = MLPClassifier(
        hidden_layer_sizes=chosen_architecture,
        max_iter=1000,
        random_state=42,
        warm_start=False)
    
    # Fit the model once to initialize weights 
    model.fit(X, y)
    
    # Calculate dimensions from the fitted MLPClassifier's actual parameter shapes
    n_dimensions = (
        sum(coef.size for coef in model.coefs_)
        + sum(intercept.size for intercept in model.intercepts_)
    )
    return model, n_dimensions

def generate_solution(n_dimensions, initialization_method = 'uniform', low = -1, high = 1):
    """
    Produces a random weight vector of the appropriate length for the
    chosen network architecture
    """
    # 1st initialization method: uniform
    if initialization_method == 'uniform':
        return np.random.uniform(low, high, n_dimensions)
    
    # 2nd initialization method: normal
    elif initialization_method == 'normal':
        return np.random.normal(0,1, n_dimensions)
    
    else:
        raise ValueError("Invalid initialization method. Choose 'uniform' or 'normal'.")
    
def vector_to_weights(vector, mlp):
    """
    Converts a flat weight vector into the weights and biases format used by MLPClassifier
    """
    expected_dimensions = (
        sum(coef.size for coef in mlp.coefs_)
        + sum(intercept.size for intercept in mlp.intercepts_)
    )
    if len(vector) != expected_dimensions:
        raise ValueError(
            f"Vector length {len(vector)} does not match expected model "
            f"parameter count {expected_dimensions}."
        )

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
    predictions on the dataset, and returns the Matthews Correlation Coefficient
    to evaluate the predictive performance on this imbalanced dataset
    """
    # 1. Convert flat vector to weights and biases
    coefs, intercepts = vector_to_weights(vector, mlp)
    
    # 2. Inject the weights into the scikit-learn MLP model
    mlp.coefs_ = coefs
    mlp.intercepts_ = intercepts
    
    # 3. Make predictions
    pred = mlp.predict(X)

    # 4. Calculate MCC
    # MCC is highly resilient to imbalanced datasets like Parkinson's.
    # It ranges from -1 (worst) to 1 (perfect).
    mcc_score = matthews_corrcoef(y, pred)
    
    return mcc_score


def evaluate_solution(vector, mlp, X, y):
    """
    Evaluates a weight vector on a dataset and returns multiple classification metrics
    """
    coefs, intercepts = vector_to_weights(vector, mlp)
    mlp.coefs_ = coefs
    mlp.intercepts_ = intercepts
    pred = mlp.predict(X)

    return {
        'weighted_f1': f1_score(y, pred, average='weighted', zero_division=0),
        'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'confusion_matrix': confusion_matrix(y, pred).tolist(),
    }
