import numpy as np
from sklearn.base import clone

class VotingClassifier:
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting
        self.estimators_ = []

    def fit(self, X, y):
        self.estimators_ = [clone(clf).fit(X, y) for _, clf in self.estimators]
        return self

    def predict(self, X):
        if self.voting == 'hard':
            predictions = np.asarray([clf.predict(X) for clf in self.estimators_]).T
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
        else:
            # Soft voting implementation
            probas = np.asarray([clf.predict_proba(X) for clf in self.estimators_])
            avg_proba = np.average(probas, axis=0)
            return np.argmax(avg_proba, axis=1)
