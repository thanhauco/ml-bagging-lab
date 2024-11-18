import itertools
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from src.metrics.classification import accuracy_score

class GridSearch:
    def __init__(self, estimator, param_grid, cv=3):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = -np.inf
        
    def fit(self, X, y):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        kf = KFold(n_splits=self.cv)
        
        for params in combinations:
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = clone(self.estimator)
                for k, v in params.items():
                    setattr(model, k, v)
                    
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))
                
            avg_score = np.mean(scores)
            if avg_score > self.best_score_:
                self.best_score_ = avg_score
                self.best_params_ = params
                
        return self
