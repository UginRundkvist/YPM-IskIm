# compute_cost_multi.py
import numpy as np

def compute_cost_multi(X: np.ndarray, y: np.ndarray, theta: np.ndarray, mode: str = 'vectorized') -> float:
    """
    Вычисляет функцию стоимости J(θ) для многомерной линейной регрессии.
    Поддерживаемые режимы:
      - 'vectorized' (по умолчанию): J = (1/(2m)) * (Xθ - y)^T (Xθ - y)
      - 'numpy':        то же через np.sum
      - 'loops':        учебная циклическая версия
    """
    m = y.shape[0]

    if mode == 'vectorized':
        errors = X @ theta - y
        return (errors @ errors) / (2 * m)

    elif mode == 'numpy':
        errors = X @ theta - y
        return float(np.sum(errors ** 2) / (2 * m))

    elif mode == 'loops':
        J = 0.0
        for i in range(m):
            hi = float(np.dot(X[i], theta))
            J += (hi - y[i]) ** 2
        return J / (2 * m)

    else:
        raise ValueError("mode must be 'vectorized', 'numpy', or 'loops'")
