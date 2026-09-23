import numpy as np

USED_INPUTS = ['k_spring', 'x']

def predict(X: np.ndarray) -> np.ndarray:
    k_spring = X[:, 0]
    x = X[:, 1]
    return 1/2*k_spring*x**2
