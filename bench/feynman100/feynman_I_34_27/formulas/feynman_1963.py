import numpy as np

USED_INPUTS = ['omega', 'h']

def predict(X: np.ndarray) -> np.ndarray:
    omega = X[:, 0]
    h = X[:, 1]
    return (h/(2*np.pi))*omega
