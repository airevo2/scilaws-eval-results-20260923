import numpy as np

USED_INPUTS = ['m', 'v', 'u', 'w']

def predict(X: np.ndarray) -> np.ndarray:
    m = X[:, 0]
    v = X[:, 1]
    u = X[:, 2]
    w = X[:, 3]
    return 1/2*m*(v**2+u**2+w**2)
