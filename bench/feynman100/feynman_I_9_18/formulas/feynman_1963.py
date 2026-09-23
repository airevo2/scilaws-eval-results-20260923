import numpy as np

USED_INPUTS = ['m1', 'm2', 'G', 'x1', 'x2', 'y1', 'y2', 'z1', 'z2']

def predict(X: np.ndarray) -> np.ndarray:
    m1 = X[:, 0]
    m2 = X[:, 1]
    G = X[:, 2]
    x1 = X[:, 3]
    x2 = X[:, 4]
    y1 = X[:, 5]
    y2 = X[:, 6]
    z1 = X[:, 7]
    z2 = X[:, 8]
    return G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
