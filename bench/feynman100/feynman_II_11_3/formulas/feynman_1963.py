import numpy as np

USED_INPUTS = ['q', 'Ef', 'm', 'omega_0', 'omega']

def predict(X: np.ndarray) -> np.ndarray:
    q = X[:, 0]
    Ef = X[:, 1]
    m = X[:, 2]
    omega_0 = X[:, 3]
    omega = X[:, 4]
    return q*Ef/(m*(omega_0**2-omega**2))
