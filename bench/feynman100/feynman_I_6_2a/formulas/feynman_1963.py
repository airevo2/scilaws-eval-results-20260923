import numpy as np

USED_INPUTS = ['theta']

def predict(X: np.ndarray) -> np.ndarray:
    theta = X[:, 0]
    return np.exp(-theta**2/2)/np.sqrt(2*np.pi)
