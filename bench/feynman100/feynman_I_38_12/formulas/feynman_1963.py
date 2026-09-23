import numpy as np

USED_INPUTS = ['m', 'q', 'h', 'epsilon']

def predict(X: np.ndarray) -> np.ndarray:
    m = X[:, 0]
    q = X[:, 1]
    h = X[:, 2]
    epsilon = X[:, 3]
    return 4*np.pi*epsilon*(h/(2*np.pi))**2/(m*q**2)
