import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from src.utils.sampling import bootstrap_sample

class BaggingClassifier:
    """
    Bagging Classifier implementation from scratch.
    
    Parameters:
        base_estimator (object): The base estimator to fit on random subsets of the dataset.
        n_estimators (int): The number of base estimators in the ensemble.
        n_jobs (int): The number of jobs to run in parallel for fit. -1 means using all processors.
        random_state (int): Seed for reproducibility.
    """
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, 
                 bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=None):
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimators_ = []
        self.estimators_features_ = []
        
    def _fit_estimator(self, estimator, X, y, seed):
        """Fit a single estimator on a bootstrap sample."""
        rng = np.random.RandomState(seed)
        
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
            
        # Sample instances
        n_samples = X.shape[0]
        if isinstance(self.max_samples, float):
            n_samples_subset = max(1, int(self.max_samples * n_samples))
        else:
            n_samples_subset = self.max_samples
            
        if self.bootstrap:
            indices = rng.choice(n_samples, n_samples_subset, replace=True)
        else:
            indices = rng.choice(n_samples, n_samples_subset, replace=False)
            
        X_subset = X[indices][:, features]
        y_subset = y[indices]
        
        estimator.fit(X_subset, y_subset)
        return estimator, features

    def fit(self, X, y):
        """
        Build a Bagging ensemble of estimators from the training set (X, y).
        """
        self.estimators_ = []
        self.estimators_features_ = []
        
        # Generate seeds for each estimator
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 2**32 - 1, size=self.n_estimators)
        
        # Parallel training
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_estimator)(
                clone(self.base_estimator), X, y, seed
            ) for seed in seeds
        )
        
        self.estimators_, self.estimators_features_ = zip(*results)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """
        Predict class for X.
        """
        # Collect predictions from all estimators
        predictions = []
        for estimator, features in zip(self.estimators_, self.estimators_features_):
            # Select the same features used for training
            X_subset = X[:, features]
            predictions.append(estimator.predict(X_subset))
            
        predictions = np.array(predictions)
        
        # Majority vote
        def majority_vote(sample_predictions):
            return np.bincount(sample_predictions, minlength=len(self.classes_)).argmax()
            
        return np.apply_along_axis(majority_vote, axis=0, arr=predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        """
        probas = []
        for estimator, features in zip(self.estimators_, self.estimators_features_):
            X_subset = X[:, features]
            probas.append(estimator.predict_proba(X_subset))
            
        return np.mean(probas, axis=0)

    @property
    def oob_score_(self):
        return 0.0 # Placeholder for real implementation
