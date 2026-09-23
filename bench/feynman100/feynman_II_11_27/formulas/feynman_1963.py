import numpy as np

USED_INPUTS = ['n', 'alpha', 'epsilon', 'Ef']

def predict(X: np.ndarray) -> np.ndarray:
    n = X[:, 0]
    alpha = X[:, 1]
    epsilon = X[:, 2]
    Ef = X[:, 3]
    return n*alpha/(1-(n*alpha/3))*epsilon*Ef
