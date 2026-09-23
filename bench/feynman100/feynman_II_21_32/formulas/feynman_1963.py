import numpy as np

USED_INPUTS = ['q', 'epsilon', 'r', 'v', 'c']

def predict(X: np.ndarray) -> np.ndarray:
    q = X[:, 0]
    epsilon = X[:, 1]
    r = X[:, 2]
    v = X[:, 3]
    c = X[:, 4]
    return q/(4*np.pi*epsilon*r*(1-v/c))
