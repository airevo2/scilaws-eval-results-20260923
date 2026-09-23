import numpy as np

USED_INPUTS = ['x', 'c', 'u', 't']

def predict(X: np.ndarray) -> np.ndarray:
    x = X[:, 0]
    c = X[:, 1]
    u = X[:, 2]
    t = X[:, 3]
    return (t-u*x/c**2)/np.sqrt(1-u**2/c**2)
