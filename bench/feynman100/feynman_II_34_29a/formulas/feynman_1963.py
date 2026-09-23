import numpy as np

USED_INPUTS = ['q', 'h', 'm']

def predict(X: np.ndarray) -> np.ndarray:
    q = X[:, 0]
    h = X[:, 1]
    m = X[:, 2]
    return q*h/(4*np.pi*m)
