import numpy as np

USED_INPUTS = ['n_rho', 'mom', 'B', 'kb', 'T']

def predict(X: np.ndarray) -> np.ndarray:
    n_rho = X[:, 0]
    mom = X[:, 1]
    B = X[:, 2]
    kb = X[:, 3]
    T = X[:, 4]
    return n_rho*mom*np.tanh(mom*B/(kb*T))
