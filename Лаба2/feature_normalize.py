import numpy as np


#  Нормализует признаки по столбцам: (X - mu) / sigma.
#Возвращает кортеж: (X_norm, mu, sigma).
def feature_normalize(X: np.ndarray):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=0)
    sigma_safe = sigma.copy()
    sigma_safe[sigma_safe == 0] = 1.0
    X_norm = (X - mu) / sigma_safe
    return X_norm, mu, sigma_safe