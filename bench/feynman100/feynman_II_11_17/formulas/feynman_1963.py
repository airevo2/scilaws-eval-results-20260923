import numpy as np

USED_INPUTS = ['n_0', 'kb', 'T', 'theta', 'p_d', 'Ef']

def predict(X: np.ndarray) -> np.ndarray:
    n_0 = X[:, 0]
    kb = X[:, 1]
    T = X[:, 2]
    theta = X[:, 3]
    p_d = X[:, 4]
    Ef = X[:, 5]
    return n_0*(1+p_d*Ef*np.cos(theta)/(kb*T))
