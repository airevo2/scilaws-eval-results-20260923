import numpy as np

USED_INPUTS = ['I_0', 'q', 'Volt', 'kb', 'T']

def predict(X: np.ndarray) -> np.ndarray:
    I_0 = X[:, 0]
    q = X[:, 1]
    Volt = X[:, 2]
    kb = X[:, 3]
    T = X[:, 4]
    return I_0*(np.exp(q*Volt/(kb*T))-1)
