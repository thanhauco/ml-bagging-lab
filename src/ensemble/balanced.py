import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from src.ensemble.bagging import BaggingClassifier

class BalancedBaggingClassifier(BaggingClassifier):
    """
    Balanced Bagging Classifier.
    
    This implementation balances the bootstrap samples by undersampling the majority class
    to match the size of the minority class in each bootstrap sample.
    """
    def __init__(self, base_estimator=None, n_estimators=10, max_features=1.0, 
                 bootstrap_features=False, n_jobs=1, random_state=None):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=max_features,
            bootstrap=True,
            bootstrap_features=bootstrap_features,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
    def _fit_estimator(self, estimator, X, y, seed):
        """Fit a single estimator on a balanced bootstrap sample."""
        rng = np.random.RandomState(seed)
        
        # Identify classes and counts
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        
        n_minority = np.min(counts)
        
        # Indices for each class
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Sample with replacement from both to match minority size (or just take all minority)
        # Standard Balanced Bagging: Take all minority, sample majority to match
        
        # Sample majority indices to match minority count
        sampled_majority_indices = rng.choice(majority_indices, n_minority, replace=False)
        
        # Combine indices
        indices = np.concatenate([minority_indices, sampled_majority_indices])
        
        # Shuffle
        rng.shuffle(indices)
        
        # Sample features
        n_features = X.shape[1]
        if isinstance(self.max_features, float):
            n_features_subset = max(1, int(self.max_features * n_features))
        else:
            n_features_subset = self.max_features
            
        if n_features_subset < n_features:
            features = rng.choice(n_features, n_features_subset, replace=self.bootstrap_features)
        else:
            features = np.arange(n_features)
            
        X_subset = X[indices][:, features]
        y_subset = y[indices]
        
        estimator.fit(X_subset, y_subset)
        return estimator, features
