import numpy as np

def computeCostMulti(X, y, theta):
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    J = (1 / (2 * m)) * np.sum(errors ** 2)
    return J
