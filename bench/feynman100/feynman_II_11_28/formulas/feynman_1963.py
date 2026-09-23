import numpy as np

USED_INPUTS = ['n', 'alpha']

def predict(X: np.ndarray) -> np.ndarray:
    n = X[:, 0]
    alpha = X[:, 1]
    return 1+n*alpha/(1-(n*alpha/3))
