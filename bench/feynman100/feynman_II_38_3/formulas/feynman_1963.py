import numpy as np

USED_INPUTS = ['Y', 'A', 'd', 'x']

def predict(X: np.ndarray) -> np.ndarray:
    Y = X[:, 0]
    A = X[:, 1]
    d = X[:, 2]
    x = X[:, 3]
    return Y*A*x/d
