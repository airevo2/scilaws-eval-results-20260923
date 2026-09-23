import numpy as np

USED_INPUTS = ['rho_c_0', 'q', 'A_vec', 'm']

def predict(X: np.ndarray) -> np.ndarray:
    rho_c_0 = X[:, 0]
    q = X[:, 1]
    A_vec = X[:, 2]
    m = X[:, 3]
    return -rho_c_0*q*A_vec/m
