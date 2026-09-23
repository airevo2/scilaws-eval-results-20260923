import numpy as np

USED_INPUTS = ['n_0', 'm', 'x', 'T', 'g', 'kb']

def predict(X: np.ndarray) -> np.ndarray:
    n_0 = X[:, 0]
    m = X[:, 1]
    x = X[:, 2]
    T = X[:, 3]
    g = X[:, 4]
    kb = X[:, 5]
    return n_0*np.exp(-m*g*x/(kb*T))
