import numpy as np

USED_INPUTS = ['q', 'epsilon', 'd']

def predict(X: np.ndarray) -> np.ndarray:
    q = X[:, 0]
    epsilon = X[:, 1]
    d = X[:, 2]
    return 3/5*q**2/(4*np.pi*epsilon*d)
