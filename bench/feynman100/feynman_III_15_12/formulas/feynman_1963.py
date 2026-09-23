import numpy as np

USED_INPUTS = ['U', 'k', 'd']

def predict(X: np.ndarray) -> np.ndarray:
    U = X[:, 0]
    k = X[:, 1]
    d = X[:, 2]
    return 2*U*(1-np.cos(k*d))
