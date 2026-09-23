import numpy as np

USED_INPUTS = ['gamma', 'kb', 'A', 'v']

def predict(X: np.ndarray) -> np.ndarray:
    gamma = X[:, 0]
    kb = X[:, 1]
    A = X[:, 2]
    v = X[:, 3]
    return 1/(gamma-1)*kb*v/A
