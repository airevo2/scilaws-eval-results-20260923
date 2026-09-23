import numpy as np

USED_INPUTS = ['q2', 'Ef']

def predict(X: np.ndarray) -> np.ndarray:
    q2 = X[:, 0]
    Ef = X[:, 1]
    return q2*Ef
