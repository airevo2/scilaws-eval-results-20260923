import numpy as np

USED_INPUTS = ['n', 'T', 'V', 'kb']

def predict(X: np.ndarray) -> np.ndarray:
    n = X[:, 0]
    T = X[:, 1]
    V = X[:, 2]
    kb = X[:, 3]
    return n*kb*T/V
