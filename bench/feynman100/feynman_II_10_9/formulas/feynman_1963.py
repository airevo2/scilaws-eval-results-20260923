import numpy as np

USED_INPUTS = ['sigma_den', 'epsilon', 'chi']

def predict(X: np.ndarray) -> np.ndarray:
    sigma_den = X[:, 0]
    epsilon = X[:, 1]
    chi = X[:, 2]
    return sigma_den/epsilon*1/(1+chi)
