import numpy as np

USED_INPUTS = ['pr', 'V']

def predict(X: np.ndarray) -> np.ndarray:
    pr = X[:, 0]
    V = X[:, 1]
    return 3/2*pr*V
