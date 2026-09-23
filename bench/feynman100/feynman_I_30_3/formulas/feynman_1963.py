import numpy as np

USED_INPUTS = ['Int_0', 'theta', 'n']

def predict(X: np.ndarray) -> np.ndarray:
    Int_0 = X[:, 0]
    theta = X[:, 1]
    n = X[:, 2]
    return Int_0*np.sin(n*theta/2)**2/np.sin(theta/2)**2
