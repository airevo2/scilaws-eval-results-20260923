import numpy as np

USED_INPUTS = ['alpha', 'n', 'd']

def predict(X: np.ndarray) -> np.ndarray:
    alpha = X[:, 0]
    n = X[:, 1]
    d = X[:, 2]
    return 2*np.pi*alpha/(n*d)
