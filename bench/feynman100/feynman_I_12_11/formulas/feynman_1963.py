import numpy as np

USED_INPUTS = ['q', 'Ef', 'B', 'v', 'theta']

def predict(X: np.ndarray) -> np.ndarray:
    q = X[:, 0]
    Ef = X[:, 1]
    B = X[:, 2]
    v = X[:, 3]
    theta = X[:, 4]
    return q*(Ef+B*v*np.sin(theta))
