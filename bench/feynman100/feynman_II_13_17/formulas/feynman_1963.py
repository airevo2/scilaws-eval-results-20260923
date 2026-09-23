import numpy as np

USED_INPUTS = ['epsilon', 'c', 'I', 'r']

def predict(X: np.ndarray) -> np.ndarray:
    epsilon = X[:, 0]
    c = X[:, 1]
    I = X[:, 2]
    r = X[:, 3]
    return 1/(4*np.pi*epsilon*c**2)*2*I/r
