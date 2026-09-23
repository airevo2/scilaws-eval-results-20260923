import numpy as np

USED_INPUTS = ['mom', 'H', 'kb', 'T', 'alpha', 'epsilon', 'c', 'M']

def predict(X: np.ndarray) -> np.ndarray:
    mom = X[:, 0]
    H = X[:, 1]
    kb = X[:, 2]
    T = X[:, 3]
    alpha = X[:, 4]
    epsilon = X[:, 5]
    c = X[:, 6]
    M = X[:, 7]
    return mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M
