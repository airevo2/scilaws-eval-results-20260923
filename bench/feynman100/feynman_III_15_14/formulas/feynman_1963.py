import numpy as np

USED_INPUTS = ['h', 'E_n', 'd']

def predict(X: np.ndarray) -> np.ndarray:
    h = X[:, 0]
    E_n = X[:, 1]
    d = X[:, 2]
    return (h/(2*np.pi))**2/(2*E_n*d**2)
