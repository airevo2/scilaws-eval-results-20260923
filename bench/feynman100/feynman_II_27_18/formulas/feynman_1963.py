import numpy as np

USED_INPUTS = ['epsilon', 'Ef']

def predict(X: np.ndarray) -> np.ndarray:
    epsilon = X[:, 0]
    Ef = X[:, 1]
    return epsilon*Ef**2
