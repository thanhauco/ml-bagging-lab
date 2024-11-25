import numpy as np

class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        return np.searchsorted(self.classes_, y)

    def fit_transform(self, y):
        return self.fit(y).transform(y)
