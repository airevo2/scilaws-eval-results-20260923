import numpy as np

USED_INPUTS = ['m1', 'm2', 'r1', 'r2']

def predict(X: np.ndarray) -> np.ndarray:
    m1 = X[:, 0]
    m2 = X[:, 1]
    r1 = X[:, 2]
    r2 = X[:, 3]
    return (m1*r1+m2*r2)/(m1+m2)
