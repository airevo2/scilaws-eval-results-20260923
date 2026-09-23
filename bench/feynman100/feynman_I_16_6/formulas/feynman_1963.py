import numpy as np

USED_INPUTS = ['c', 'v', 'u']

def predict(X: np.ndarray) -> np.ndarray:
    c = X[:, 0]
    v = X[:, 1]
    u = X[:, 2]
    return (u+v)/(1+u*v/c**2)
