import numpy as np

USED_INPUTS = ['q', 'epsilon', 'r']

def predict(X: np.ndarray) -> np.ndarray:
    q = X[:, 0]
    epsilon = X[:, 1]
    r = X[:, 2]
    return q/(4*np.pi*epsilon*r)
