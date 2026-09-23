import numpy as np

USED_INPUTS = ['q', 'v', 'B', 'p']

def predict(X: np.ndarray) -> np.ndarray:
    q = X[:, 0]
    v = X[:, 1]
    B = X[:, 2]
    p = X[:, 3]
    return q*v*B/p
