import numpy as np

USED_INPUTS = ['E_n', 'd', 'k', 'h']

def predict(X: np.ndarray) -> np.ndarray:
    E_n = X[:, 0]
    d = X[:, 1]
    k = X[:, 2]
    h = X[:, 3]
    return 2*E_n*d**2*k/(h/(2*np.pi))
