import numpy as np

USED_INPUTS = ['c', 'v', 'omega_0']

def predict(X: np.ndarray) -> np.ndarray:
    c = X[:, 0]
    v = X[:, 1]
    omega_0 = X[:, 2]
    return omega_0/(1-v/c)
