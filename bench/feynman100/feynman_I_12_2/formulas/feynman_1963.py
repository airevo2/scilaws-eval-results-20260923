import numpy as np

USED_INPUTS = ['q1', 'q2', 'epsilon', 'r']

def predict(X: np.ndarray) -> np.ndarray:
    q1 = X[:, 0]
    q2 = X[:, 1]
    epsilon = X[:, 2]
    r = X[:, 3]
    return q1*q2*r/(4*np.pi*epsilon*r**3)
