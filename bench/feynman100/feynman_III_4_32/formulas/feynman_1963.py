import numpy as np

USED_INPUTS = ['h', 'omega', 'kb', 'T']

def predict(X: np.ndarray) -> np.ndarray:
    h = X[:, 0]
    omega = X[:, 1]
    kb = X[:, 2]
    T = X[:, 3]
    return 1/(np.exp((h/(2*np.pi))*omega/(kb*T))-1)
