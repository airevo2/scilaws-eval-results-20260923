import numpy as np

USED_INPUTS = ['x', 'u', 'c', 't']

def predict(X: np.ndarray) -> np.ndarray:
    x = X[:, 0]
    u = X[:, 1]
    c = X[:, 2]
    t = X[:, 3]
    return (x-u*t)/np.sqrt(1-u**2/c**2)
