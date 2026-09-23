import numpy as np

USED_INPUTS = ['p_d', 'Ef', 't', 'h', 'omega', 'omega_0']

def predict(X: np.ndarray) -> np.ndarray:
    p_d = X[:, 0]
    Ef = X[:, 1]
    t = X[:, 2]
    h = X[:, 3]
    omega = X[:, 4]
    omega_0 = X[:, 5]
    return (p_d*Ef*t/(h/(2*np.pi)))*np.sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2
