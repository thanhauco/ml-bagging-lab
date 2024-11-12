from src.metrics.classification import accuracy_score
import numpy as np
def test_acc(): assert accuracy_score(np.array([1]), np.array([1])) == 1.0