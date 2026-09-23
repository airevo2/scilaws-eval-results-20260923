import numpy as np

USED_INPUTS = ['q1', 'epsilon', 'r']

def predict(X: np.ndarray) -> np.ndarray:
    q1 = X[:, 0]
    epsilon = X[:, 1]
    r = X[:, 2]
    return q1*r/(4*np.pi*epsilon*r**3)
