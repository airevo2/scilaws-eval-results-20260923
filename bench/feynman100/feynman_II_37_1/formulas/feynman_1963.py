import numpy as np

USED_INPUTS = ['mom', 'B', 'chi']

def predict(X: np.ndarray) -> np.ndarray:
    mom = X[:, 0]
    B = X[:, 1]
    chi = X[:, 2]
    return mom*(1+chi)*B
