import numpy as np

USED_INPUTS = ['r', 'F', 'theta']

def predict(X: np.ndarray) -> np.ndarray:
    r = X[:, 0]
    F = X[:, 1]
    theta = X[:, 2]
    return r*F*np.sin(theta)
