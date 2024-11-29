import numpy as np
from sklearn.base import clone
from sklearn.tree import ExtraTreeRegressor

class IsolationForest:
    def __init__(self, n_estimators=100, max_samples='auto', random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_ = []
        self.max_samples_ = 0

    def fit(self, X):
        n_samples = X.shape[0]
        if self.max_samples == 'auto':
            self.max_samples_ = min(256, n_samples)
        else:
            self.max_samples_ = self.max_samples
            
        rng = np.random.RandomState(self.random_state)
        
        for _ in range(self.n_estimators):
            tree = ExtraTreeRegressor(max_depth=int(np.ceil(np.log2(self.max_samples_))), random_state=rng)
            indices = rng.choice(n_samples, self.max_samples_, replace=False)
            # We use X as both features and targets for unsupervised split simulation
            tree.fit(X[indices], X[indices]) 
            self.estimators_.append(tree)
            
        return self

    def decision_function(self, X):
        # Simplified path length calculation
        # In real IF, we calculate average path length. Here we use depth as proxy.
        depths = np.array([tree.apply(X) for tree in self.estimators_])
        # This is a placeholder for actual path length logic which is complex
        # We return negative mean depth (shallower = more anomalous)
        return -np.mean(depths, axis=0)

    def predict(self, X):
        scores = self.decision_function(X)
        threshold = np.percentile(scores, 10) # Assume 10% contamination
        return np.where(scores < threshold, -1, 1)
