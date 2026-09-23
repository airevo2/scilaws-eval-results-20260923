import numpy as np

USED_INPUTS = ['gamma', 'pr', 'rho']

def predict(X: np.ndarray) -> np.ndarray:
    gamma = X[:, 0]
    pr = X[:, 1]
    rho = X[:, 2]
    return np.sqrt(gamma*pr/rho)
