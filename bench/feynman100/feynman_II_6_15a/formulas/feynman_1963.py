import numpy as np

USED_INPUTS = ['epsilon', 'p_d', 'r', 'x', 'y', 'z']

def predict(X: np.ndarray) -> np.ndarray:
    epsilon = X[:, 0]
    p_d = X[:, 1]
    r = X[:, 2]
    x = X[:, 3]
    y = X[:, 4]
    z = X[:, 5]
    return p_d/(4*np.pi*epsilon)*3*z/r**5*np.sqrt(x**2+y**2)
