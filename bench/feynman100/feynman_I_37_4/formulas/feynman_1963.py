import numpy as np

USED_INPUTS = ['I1', 'I2', 'delta']

def predict(X: np.ndarray) -> np.ndarray:
    I1 = X[:, 0]
    I2 = X[:, 1]
    delta = X[:, 2]
    return I1+I2+2*np.sqrt(I1*I2)*np.cos(delta)
