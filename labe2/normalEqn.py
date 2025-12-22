import numpy as np

def normalEqn(X, y):
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta
