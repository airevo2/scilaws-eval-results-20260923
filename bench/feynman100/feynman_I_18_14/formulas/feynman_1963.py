import numpy as np

USED_INPUTS = ['m', 'r', 'v', 'theta']

def predict(X: np.ndarray) -> np.ndarray:
    m = X[:, 0]
    r = X[:, 1]
    v = X[:, 2]
    theta = X[:, 3]
    return m*r*v*np.sin(theta)
