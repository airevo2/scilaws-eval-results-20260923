import numpy as np

USED_INPUTS = ['gamma', 'pr', 'V']

def predict(X: np.ndarray) -> np.ndarray:
    gamma = X[:, 0]
    pr = X[:, 1]
    V = X[:, 2]
    return 1/(gamma-1)*pr*V
