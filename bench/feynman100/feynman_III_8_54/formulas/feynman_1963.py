import numpy as np

USED_INPUTS = ['E_n', 't', 'h']

def predict(X: np.ndarray) -> np.ndarray:
    E_n = X[:, 0]
    t = X[:, 1]
    h = X[:, 2]
    return np.sin(E_n*t/(h/(2*np.pi)))**2
