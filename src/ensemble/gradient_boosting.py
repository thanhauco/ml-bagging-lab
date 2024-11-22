import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        self.initial_prediction_ = 0.0

    def fit(self, X, y):
        self.initial_prediction_ = np.mean(y)
        y_pred = np.full_like(y, self.initial_prediction_)
        
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, residuals)
            
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.estimators_.append(tree)
            
        return self

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction_)
        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
