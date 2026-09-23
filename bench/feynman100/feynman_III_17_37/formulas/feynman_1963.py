import numpy as np

USED_INPUTS = ['beta', 'alpha', 'theta']

def predict(X: np.ndarray) -> np.ndarray:
    beta = X[:, 0]
    alpha = X[:, 1]
    theta = X[:, 2]
    return beta*(1+alpha*np.cos(theta))
