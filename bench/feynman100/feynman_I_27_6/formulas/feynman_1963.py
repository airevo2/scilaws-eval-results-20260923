import numpy as np

USED_INPUTS = ['d1', 'd2', 'n']

def predict(X: np.ndarray) -> np.ndarray:
    d1 = X[:, 0]
    d2 = X[:, 1]
    n = X[:, 2]
    return 1/(1/d1+n/d2)
