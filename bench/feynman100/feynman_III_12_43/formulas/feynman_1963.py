import numpy as np

USED_INPUTS = ['n', 'h']

def predict(X: np.ndarray) -> np.ndarray:
    n = X[:, 0]
    h = X[:, 1]
    return n*(h/(2*np.pi))
