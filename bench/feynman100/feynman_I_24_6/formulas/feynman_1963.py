import numpy as np

USED_INPUTS = ['m', 'omega', 'omega_0', 'x']

def predict(X: np.ndarray) -> np.ndarray:
    m = X[:, 0]
    omega = X[:, 1]
    omega_0 = X[:, 2]
    x = X[:, 3]
    return 1/2*m*(omega**2+omega_0**2)*1/2*x**2
