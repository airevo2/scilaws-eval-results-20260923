import numpy as np

USED_INPUTS = ['mom', 'B', 'h']

def predict(X: np.ndarray) -> np.ndarray:
    mom = X[:, 0]
    B = X[:, 1]
    h = X[:, 2]
    return 2*mom*B/(h/(2*np.pi))
