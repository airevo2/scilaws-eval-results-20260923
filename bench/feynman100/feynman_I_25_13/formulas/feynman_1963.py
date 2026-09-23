import numpy as np

USED_INPUTS = ['q', 'C']

def predict(X: np.ndarray) -> np.ndarray:
    q = X[:, 0]
    C = X[:, 1]
    return q/C
