import numpy as np
from computeCostMulti import computeCostMulti

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        theta = theta - (alpha / m) * (X.T @ errors) #вычисляет градиент функции стоимости по всем параметрам
        J_history[i] = computeCostMulti(X, y, theta)

    return theta, J_history
