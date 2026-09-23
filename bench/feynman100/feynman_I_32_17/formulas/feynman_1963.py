import numpy as np

USED_INPUTS = ['epsilon', 'c', 'Ef', 'r', 'omega', 'omega_0']

def predict(X: np.ndarray) -> np.ndarray:
    epsilon = X[:, 0]
    c = X[:, 1]
    Ef = X[:, 2]
    r = X[:, 3]
    omega = X[:, 4]
    omega_0 = X[:, 5]
    return (1/2*epsilon*c*Ef**2)*(8*np.pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)
