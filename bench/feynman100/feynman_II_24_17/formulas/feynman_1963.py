import numpy as np

USED_INPUTS = ['omega', 'c', 'd']

def predict(X: np.ndarray) -> np.ndarray:
    omega = X[:, 0]
    c = X[:, 1]
    d = X[:, 2]
    return np.sqrt(omega**2/c**2-np.pi**2/d**2)
