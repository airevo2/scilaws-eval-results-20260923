import numpy as np

USED_INPUTS = ['mu_drift', 'q', 'Volt', 'd']

def predict(X: np.ndarray) -> np.ndarray:
    mu_drift = X[:, 0]
    q = X[:, 1]
    Volt = X[:, 2]
    d = X[:, 3]
    return mu_drift*q*Volt/d
