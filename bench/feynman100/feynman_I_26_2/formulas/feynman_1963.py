import numpy as np

USED_INPUTS = ['n', 'theta2']

def predict(X: np.ndarray) -> np.ndarray:
    n = X[:, 0]
    theta2 = X[:, 1]
    return np.arcsin(n*np.sin(theta2))
