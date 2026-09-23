import numpy as np

USED_INPUTS = ['n_rho', 'p_d', 'Ef', 'kb', 'T']

def predict(X: np.ndarray) -> np.ndarray:
    n_rho = X[:, 0]
    p_d = X[:, 1]
    Ef = X[:, 2]
    kb = X[:, 3]
    T = X[:, 4]
    return n_rho*p_d**2*Ef/(3*kb*T)
