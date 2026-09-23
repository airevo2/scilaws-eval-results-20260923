import numpy as np

USED_INPUTS = ['m', 'q', 'h', 'n', 'epsilon']

def predict(X: np.ndarray) -> np.ndarray:
    m = X[:, 0]
    q = X[:, 1]
    h = X[:, 2]
    n = X[:, 3]
    epsilon = X[:, 4]
    return -m*q**4/(2*(4*np.pi*epsilon)**2*(h/(2*np.pi))**2)*(1/n**2)
