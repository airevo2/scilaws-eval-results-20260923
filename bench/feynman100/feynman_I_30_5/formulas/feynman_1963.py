import numpy as np

USED_INPUTS = ['lambd', 'd', 'n']

def predict(X: np.ndarray) -> np.ndarray:
    lambd = X[:, 0]
    d = X[:, 1]
    n = X[:, 2]
    return np.arcsin(lambd/(n*d))
