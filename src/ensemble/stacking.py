import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from src.linear_model.logistic import LogisticRegression

class StackingClassifier:
    def __init__(self, estimators, final_estimator=None, cv=5):
        self.estimators = estimators
        self.final_estimator = final_estimator if final_estimator else LogisticRegression()
        self.cv = cv
        self.estimators_ = []
        self.final_estimator_ = None

    def fit(self, X, y):
        self.estimators_ = [clone(clf).fit(X, y) for _, clf in self.estimators]
        
        # Generate meta-features
        meta_features = np.zeros((X.shape[0], len(self.estimators)))
        kf = KFold(n_splits=self.cv)
        
        for i, (_, clf) in enumerate(self.estimators):
            for train_idx, val_idx in kf.split(X):
                instance = clone(clf).fit(X[train_idx], y[train_idx])
                meta_features[val_idx, i] = instance.predict(X[val_idx])
                
        self.final_estimator_ = clone(self.final_estimator).fit(meta_features, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([clf.predict(X) for clf in self.estimators_])
        return self.final_estimator_.predict(meta_features)
