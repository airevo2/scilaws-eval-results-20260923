import numpy as np

USED_INPUTS = ['epsilon', 'c', 'Ef']

def predict(X: np.ndarray) -> np.ndarray:
    epsilon = X[:, 0]
    c = X[:, 1]
    Ef = X[:, 2]
    return epsilon*c*Ef**2
