import numpy as np
from compute_cost_multi import compute_cost_multi

def gradient_descent_multi(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    alpha: float,
    num_iters: int,
    mode: str = 'vectorized'
):
    """
    Градиентный спуск для многомерного случая.
    Режимы:
      - 'vectorized': θ := θ - (α/m) X^T (Xθ - y)
      - 'numpy':      градиент через суммирование np.sum
      - 'loops':      учебная цикличная версия
    Возвращает: (theta, J_history)
    """
    m, n = X.shape
    theta = theta.copy()
    J_history = np.zeros(num_iters, dtype=float)

    for it in range(num_iters):
        if mode == 'vectorized':
            errors = X @ theta - y
            grad = (X.T @ errors) / m

        elif mode == 'numpy':
            errors = X @ theta - y
            # эквивалентно (X.T @ errors)/m, но через суммирование:
            grad = np.sum(errors[:, None] * X, axis=0) / m

        elif mode == 'loops':
            # посчитаем errors один раз
            errors = X @ theta - y
            grad = np.zeros(n, dtype=float)
            for j in range(n):
                s = 0.0
                for i in range(m):
                    s += errors[i] * X[i, j]
                grad[j] = s / m
        else:
            raise ValueError("mode must be 'vectorized', 'numpy', or 'loops'")

        theta -= alpha * grad
        # для истории J используем точную векторную формулу
        J_history[it] = compute_cost_multi(X, y, theta, mode='vectorized')

    return theta, J_history