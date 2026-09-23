import numpy as np

USED_INPUTS = ['x1', 'omega', 't', 'alpha']

def predict(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    omega = X[:, 1]
    t = X[:, 2]
    alpha = X[:, 3]
    return x1*(np.cos(omega*t)+alpha*np.cos(omega*t)**2)
