import numpy as np

#Аналитическое решение нормальных уравнений:
#θ = (X^T X)^(-1) X^T y
def normal_eqn(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(X.T @ X) @ (X.T @ y)