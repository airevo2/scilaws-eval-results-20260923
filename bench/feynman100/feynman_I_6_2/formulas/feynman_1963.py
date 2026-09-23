import numpy as np

USED_INPUTS = ['sigma', 'theta']

def predict(X: np.ndarray) -> np.ndarray:
    sigma = X[:, 0]
    theta = X[:, 1]
    return np.exp(-(theta/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)
