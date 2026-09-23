import numpy as np

USED_INPUTS = ['p_d', 'Ef', 'theta']

def predict(X: np.ndarray) -> np.ndarray:
    p_d = X[:, 0]
    Ef = X[:, 1]
    theta = X[:, 2]
    return -p_d*Ef*np.cos(theta)
