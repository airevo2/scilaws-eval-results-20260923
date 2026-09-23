import numpy as np

USED_INPUTS = ['Pwr', 'r']

def predict(X: np.ndarray) -> np.ndarray:
    Pwr = X[:, 0]
    r = X[:, 1]
    return Pwr/(4*np.pi*r**2)
