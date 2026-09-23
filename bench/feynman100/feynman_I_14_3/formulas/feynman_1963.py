import numpy as np

USED_INPUTS = ['m', 'g', 'z']

def predict(X: np.ndarray) -> np.ndarray:
    m = X[:, 0]
    g = X[:, 1]
    z = X[:, 2]
    return m*g*z
