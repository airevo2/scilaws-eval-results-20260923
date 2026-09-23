import numpy as np

USED_INPUTS = ['n', 'kb', 'T', 'V1', 'V2']

def predict(X: np.ndarray) -> np.ndarray:
    n = X[:, 0]
    kb = X[:, 1]
    T = X[:, 2]
    V1 = X[:, 3]
    V2 = X[:, 4]
    return n*kb*T*np.log(V2/V1)
