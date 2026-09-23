import numpy as np

USED_INPUTS = ['q', 'v', 'r']

def predict(X: np.ndarray) -> np.ndarray:
    q = X[:, 0]
    v = X[:, 1]
    r = X[:, 2]
    return q*v*r/2
