import numpy as np

USED_INPUTS = ['x1', 'x2', 'theta1', 'theta2']

def predict(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    x2 = X[:, 1]
    theta1 = X[:, 2]
    theta2 = X[:, 3]
    return np.sqrt(x1**2+x2**2-2*x1*x2*np.cos(theta1-theta2))
