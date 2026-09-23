import numpy as np

USED_INPUTS = ['omega', 'c']

def predict(X: np.ndarray) -> np.ndarray:
    omega = X[:, 0]
    c = X[:, 1]
    return omega/c
