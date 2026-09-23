import numpy as np

USED_INPUTS = ['omega', 'T', 'h', 'kb', 'c']

def predict(X: np.ndarray) -> np.ndarray:
    omega = X[:, 0]
    T = X[:, 1]
    h = X[:, 2]
    kb = X[:, 3]
    c = X[:, 4]
    return h/(2*np.pi)*omega**3/(np.pi**2*c**2*(np.exp((h/(2*np.pi))*omega/(kb*T))-1))
