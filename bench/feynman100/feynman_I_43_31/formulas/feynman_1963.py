import numpy as np

USED_INPUTS = ['mob', 'T', 'kb']

def predict(X: np.ndarray) -> np.ndarray:
    mob = X[:, 0]
    T = X[:, 1]
    kb = X[:, 2]
    return mob*kb*T
