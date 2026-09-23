import numpy as np

USED_INPUTS = ['mom', 'Bx', 'By', 'Bz']

def predict(X: np.ndarray) -> np.ndarray:
    mom = X[:, 0]
    Bx = X[:, 1]
    By = X[:, 2]
    Bz = X[:, 3]
    return mom*np.sqrt(Bx**2+By**2+Bz**2)
