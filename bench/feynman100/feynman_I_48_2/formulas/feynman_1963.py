import numpy as np

USED_INPUTS = ['m', 'v', 'c']

def predict(X: np.ndarray) -> np.ndarray:
    m = X[:, 0]
    v = X[:, 1]
    c = X[:, 2]
    return m*c**2/np.sqrt(1-v**2/c**2)
