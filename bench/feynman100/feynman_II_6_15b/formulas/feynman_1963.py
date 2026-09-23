import numpy as np

USED_INPUTS = ['epsilon', 'p_d', 'theta', 'r']

def predict(X: np.ndarray) -> np.ndarray:
    epsilon = X[:, 0]
    p_d = X[:, 1]
    theta = X[:, 2]
    r = X[:, 3]
    return p_d/(4*np.pi*epsilon)*3*np.cos(theta)*np.sin(theta)/r**3
