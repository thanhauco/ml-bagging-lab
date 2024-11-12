import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from src.utils.sampling import bootstrap_sample

class BaggingRegressor:
    def __init__(self, base_estimator=None, n_estimators=10, n_jobs=1, random_state=None):
        self.base_estimator = base_estimator if base_estimator else DecisionTreeRegressor()
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimators_ = []
        
    def _fit_estimator(self, estimator, X, y, seed):
        X_sample, y_sample = bootstrap_sample(X, y, random_state=seed)
        estimator.fit(X_sample, y_sample)
        return estimator

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 2**32 - 1, size=self.n_estimators)
        
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_estimator)(
                clone(self.base_estimator), X, y, seed
            ) for seed in seeds
        )
        return self

    def predict(self, X):
        predictions = np.array([est.predict(X) for est in self.estimators_])
        return np.mean(predictions, axis=0)
