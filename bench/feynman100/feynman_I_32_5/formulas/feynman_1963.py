import numpy as np

USED_INPUTS = ['q', 'a', 'epsilon', 'c']

def predict(X: np.ndarray) -> np.ndarray:
    q = X[:, 0]
    a = X[:, 1]
    epsilon = X[:, 2]
    c = X[:, 3]
    return q**2*a**2/(6*np.pi*epsilon*c**3)
