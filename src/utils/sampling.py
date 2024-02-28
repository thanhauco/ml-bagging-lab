import numpy as np

def bootstrap_sample(X, y, random_state=None):
    """
    Generate a bootstrap sample from the dataset (Sampling with replacement).
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        random_state (int, optional): Seed for reproducibility.
        
    Returns:
        tuple: (X_sample, y_sample)
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]

def feature_sample(X, max_features, random_state=None):
    """
    Select a random subset of features.
    
    Args:
        X (np.ndarray): Feature matrix.
        max_features (int or float): Number of features to select.
            If float, int(max_features * n_features) features are considered.
        random_state (int, optional): Seed for reproducibility.
        
    Returns:
        tuple: (X_subset, selected_indices)
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_features = X.shape[1]
    
    if isinstance(max_features, float):
        n_selected = max(1, int(max_features * n_features))
    else:
        n_selected = max_features
        
    selected_indices = np.random.choice(n_features, size=n_selected, replace=False)
    return X[:, selected_indices], selected_indices
