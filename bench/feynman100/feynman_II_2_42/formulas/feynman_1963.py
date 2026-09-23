import numpy as np

USED_INPUTS = ['kappa', 'T1', 'T2', 'A', 'd']

def predict(X: np.ndarray) -> np.ndarray:
    kappa = X[:, 0]
    T1 = X[:, 1]
    T2 = X[:, 2]
    A = X[:, 3]
    d = X[:, 4]
    return kappa*(T2-T1)*A/d
