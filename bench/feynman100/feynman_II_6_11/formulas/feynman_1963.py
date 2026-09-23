import numpy as np

USED_INPUTS = ['epsilon', 'p_d', 'theta', 'r']

def predict(X: np.ndarray) -> np.ndarray:
    epsilon = X[:, 0]
    p_d = X[:, 1]
    theta = X[:, 2]
    r = X[:, 3]
    return 1/(4*np.pi*epsilon)*p_d*np.cos(theta)/r**2
