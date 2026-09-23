import numpy as np

USED_INPUTS = ['mu', 'Nn']

def predict(X: np.ndarray) -> np.ndarray:
    mu = X[:, 0]
    Nn = X[:, 1]
    return mu*Nn
