import numpy as np

USED_INPUTS = ['m_0', 'v', 'c']

def predict(X: np.ndarray) -> np.ndarray:
    m_0 = X[:, 0]
    v = X[:, 1]
    c = X[:, 2]
    return m_0/np.sqrt(1-v**2/c**2)
