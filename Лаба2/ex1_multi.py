import os
import sys
import numpy as np

from feature_normalize import feature_normalize
from compute_cost_multi import compute_cost_multi
from gradient_descent_multi import gradient_descent_multi
from normal_eqn import normal_eqn

# Попробуем подключить матплотлиб для визуализации (опционально)
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


def load_data_flex(path: str) -> np.ndarray:
    """
    Гибкая загрузка данных: пытаемся читать как CSV, затем как whitespace.
    Ожидаются 3 колонки: RPM, gears, price
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    # Попытка 1: запятая
    try:
        data = np.loadtxt(path, delimiter=',')
        if data.ndim == 1:
            # если одна строка — это не наш случай, попробуем пробел
            raise ValueError
        return data.astype(float)
    except Exception:
        pass

    # Попытка 2: пробельный разделитель (любое количество пробелов/табов)
    data = np.loadtxt(path)
    if data.ndim == 1:
        raise ValueError("Не удалось распознать формат файла. Требуется минимум 2 строки.")
    return data.astype(float)


def main():
    # Настройки
    data_path = 'ex1data2.txt'  # ожидается в текущей папке
    # Режимы вычислений для учебных целей: 'vectorized' | 'numpy' | 'loops'
    MODE_COST = 'vectorized'
    MODE_GD = 'vectorized'

    # Сетка скоростей обучения для подбора
    alphas = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3], dtype=float)
    num_iters = 500

    print("Загрузка данных ...")
    data = load_data_flex(data_path)
    # Ожидаем 3 колонки: RPM, gears, price
    if data.shape[1] < 3:
        print(f"Ожидается как минимум 3 столбца, получено: {data.shape[1]}")
        sys.exit(1)

    X = data[:, 0:2]
    y = data[:, 2].astype(float)
    m = y.shape[0]

    # 1) Нормализация признаков (только для GD)
    print("Нормализация признаков ...")
    X_norm, mu, sigma = feature_normalize(X)
    Xn = np.c_[np.ones(m), X_norm]  # добавляем x0 = 1

    # 2) Подбор скорости обучения alpha
    print("Запуск градиентного спуска для разных alpha ...")
    n = Xn.shape[1]
    thetas = np.zeros((n, alphas.size), dtype=float)
    J_end = np.zeros(alphas.size, dtype=float)
    J_hist_all = np.zeros((num_iters, alphas.size), dtype=float)

    for i, a in enumerate(alphas):
        theta0 = np.zeros(n, dtype=float)
        theta_i, J_history_i = gradient_descent_multi(Xn, y, theta0, a, num_iters, mode=MODE_GD)
        thetas[:, i] = theta_i
        J_hist_all[:, i] = J_history_i
        J_end[i] = J_history_i[-1]
        print(f"alpha = {a:.4f}, J(final) = {J_end[i]:.6f}")

    best_idx = int(np.argmin(J_end))
    best_alpha = float(alphas[best_idx])
    theta_gd = thetas[:, best_idx]
    bestJ = float(J_end[best_idx])

    print(f"\nЛучшая скорость обучения: alpha = {best_alpha:.4f} с J(final) = {bestJ:.6f}")

    # 3) Визуализация сходимости (если доступен matplotlib)
    if HAS_PLT:
        plt.figure(figsize=(8, 5))
        for i, a in enumerate(alphas):
            plt.plot(np.arange(1, num_iters + 1), J_hist_all[:, i], label=f"alpha={a:.3f}", linewidth=1.5)
        plt.xlabel("Итерации")
        plt.ylabel("J(θ)")
        plt.title("Сходимость градиентного спуска для разных α")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    else:
        print("(matplotlib не найден — график не будет показан)")

    # 4) Аналитическое решение (нормальные уравнения) — без нормализации
    X_ne = np.c_[np.ones(m), X]
    theta_ne = normal_eqn(X_ne, y)

    # 5) Сравнение значений функции стоимости на обучающей выборке
    J_gd = compute_cost_multi(Xn, y, theta_gd, mode=MODE_COST)   # GD на нормализованных Xn
    J_ne = compute_cost_multi(X_ne, y, theta_ne, mode=MODE_COST)  # NE на исходных X

    print("\nСравнение на обучающей выборке:")
    print(f"  Градиентный спуск: J = {J_gd:.6f} (alpha={best_alpha:.4f}, iters={num_iters})")
    print(f"  Нормальные уравнения: J = {J_ne:.6f}")

    # 6) Пример предсказания для нового трактора
    RPM_new = 2000.0
    gears_new = 6.0

    x_new_norm = (np.array([RPM_new, gears_new]) - mu) / sigma
    x_new_n = np.r_[1.0, x_new_norm]
    price_pred_gd = float(x_new_n @ theta_gd)

    x_new_ne = np.array([1.0, RPM_new, gears_new])
    price_pred_ne = float(x_new_ne @ theta_ne)

    print(f"\nПример предсказания для RPM={RPM_new:g}, передач={gears_new:g}:")
    print(f"  GD:  price ≈ {price_pred_gd:.2f}")
    print(f"  NE:  price ≈ {price_pred_ne:.2f}")

    # 7) Вывод векторов параметров
    np.set_printoptions(precision=4, suppress=True)
    print("\nПараметры θ (градиентный спуск):")
    print(theta_gd)
    print("\nПараметры θ (нормальные уравнения):")
    print(theta_ne)

    print("\nГотово.")


if __name__ == "__main__":
    main()