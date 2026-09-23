import numpy as np

USED_INPUTS = ['x1', 'x2', 'y1', 'y2']

def predict(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    x2 = X[:, 1]
    y1 = X[:, 2]
    y2 = X[:, 3]
    return np.sqrt((x2-x1)**2+(y2-y1)**2)
