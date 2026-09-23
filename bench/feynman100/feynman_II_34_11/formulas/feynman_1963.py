import numpy as np

USED_INPUTS = ['g_', 'q', 'B', 'm']

def predict(X: np.ndarray) -> np.ndarray:
    g_ = X[:, 0]
    q = X[:, 1]
    B = X[:, 2]
    m = X[:, 3]
    return g_*q*B/(2*m)
