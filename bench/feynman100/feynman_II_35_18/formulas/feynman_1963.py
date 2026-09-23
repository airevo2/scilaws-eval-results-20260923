import numpy as np

USED_INPUTS = ['n_0', 'kb', 'T', 'mom', 'B']

def predict(X: np.ndarray) -> np.ndarray:
    n_0 = X[:, 0]
    kb = X[:, 1]
    T = X[:, 2]
    mom = X[:, 3]
    B = X[:, 4]
    return n_0/(np.exp(mom*B/(kb*T))+np.exp(-mom*B/(kb*T)))
