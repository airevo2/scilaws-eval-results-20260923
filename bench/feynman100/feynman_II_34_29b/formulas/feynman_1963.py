import numpy as np

USED_INPUTS = ['g_', 'h', 'Jz', 'mom', 'B']

def predict(X: np.ndarray) -> np.ndarray:
    g_ = X[:, 0]
    h = X[:, 1]
    Jz = X[:, 2]
    mom = X[:, 3]
    B = X[:, 4]
    return g_*mom*B*Jz/(h/(2*np.pi))
