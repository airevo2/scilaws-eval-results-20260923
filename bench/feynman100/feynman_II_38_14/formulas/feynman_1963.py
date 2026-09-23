import numpy as np

USED_INPUTS = ['Y', 'sigma']

def predict(X: np.ndarray) -> np.ndarray:
    Y = X[:, 0]
    sigma = X[:, 1]
    return Y/(2*(1+sigma))
