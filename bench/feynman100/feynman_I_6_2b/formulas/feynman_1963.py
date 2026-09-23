import numpy as np

USED_INPUTS = ['sigma', 'theta', 'theta1']

def predict(X: np.ndarray) -> np.ndarray:
    sigma = X[:, 0]
    theta = X[:, 1]
    theta1 = X[:, 2]
    return np.exp(-((theta-theta1)/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)
