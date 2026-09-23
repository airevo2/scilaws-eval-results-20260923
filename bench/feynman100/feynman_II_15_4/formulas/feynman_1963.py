import numpy as np

USED_INPUTS = ['mom', 'B', 'theta']

def predict(X: np.ndarray) -> np.ndarray:
    mom = X[:, 0]
    B = X[:, 1]
    theta = X[:, 2]
    return -mom*B*np.cos(theta)
