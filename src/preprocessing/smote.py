import numpy as np
from sklearn.neighbors import NearestNeighbors

class SMOTE:
    def __init__(self, k_neighbors=5, random_state=None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        
        # Identify minority class
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        
        X_minority = X[y == minority_class]
        n_minority = len(X_minority)
        n_majority = np.max(counts)
        n_synthetic = n_majority - n_minority
        
        if n_synthetic <= 0:
            return X, y
            
        neigh = NearestNeighbors(n_neighbors=self.k_neighbors)
        neigh.fit(X_minority)
        
        synthetic_samples = []
        
        for _ in range(n_synthetic):
            # Pick random minority sample
            idx = rng.randint(0, n_minority)
            sample = X_minority[idx]
            
            # Find neighbors
            nn = neigh.kneighbors([sample], return_distance=False)
            neighbor_idx = rng.choice(nn[0])
            neighbor = X_minority[neighbor_idx]
            
            # Interpolate
            diff = neighbor - sample
            gap = rng.random()
            new_sample = sample + gap * diff
            synthetic_samples.append(new_sample)
            
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(n_synthetic, minority_class)
        
        return np.vstack([X, X_synthetic]), np.hstack([y, y_synthetic])
