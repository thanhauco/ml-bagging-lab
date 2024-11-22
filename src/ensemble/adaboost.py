import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=None):
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            clf = clone(self.base_estimator)
            clf.fit(X, y, sample_weight=w)
            y_pred = clf.predict(X)
            
            incorrect = y_pred != y
            error = np.average(incorrect, weights=w, axis=0)
            
            if error >= 0.5:
                break
                
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / max(error, 1e-10))
            
            w *= np.exp(-alpha * y * y_pred) # Assuming y in {-1, 1}
            w /= np.sum(w)
            
            self.estimators_.append(clf)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)
            
        return self

    def predict(self, X):
        clf_preds = np.array([clf.predict(X) for clf in self.estimators_])
        return np.sign(np.dot(self.estimator_weights_, clf_preds))
