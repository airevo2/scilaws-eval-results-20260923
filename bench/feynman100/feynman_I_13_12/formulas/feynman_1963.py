import numpy as np

USED_INPUTS = ['m1', 'm2', 'r1', 'r2', 'G']

def predict(X: np.ndarray) -> np.ndarray:
    m1 = X[:, 0]
    m2 = X[:, 1]
    r1 = X[:, 2]
    r2 = X[:, 3]
    G = X[:, 4]
    return G*m1*m2*(1/r2-1/r1)
