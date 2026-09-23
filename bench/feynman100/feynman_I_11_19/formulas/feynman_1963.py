import numpy as np

USED_INPUTS = ['x1', 'x2', 'x3', 'y1', 'y2', 'y3']

def predict(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    y1 = X[:, 3]
    y2 = X[:, 4]
    y3 = X[:, 5]
    return x1*y1+x2*y2+x3*y3
