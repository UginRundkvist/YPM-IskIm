import numpy as np
import matplotlib.pyplot as plt

#определение линейной модели
def h(x, theta1):
    return theta1 * x

# рассчет стоимоти ошибки
def compute_cost(y_true, theta1):
    predictions = h(x, theta1)
    return (1 / (2 * m)) * np.sum((predictions - y_true) ** 2)

# без шума
def analyze_clean_data(x, y):
    theta1_values = np.linspace(0, 2, 100)
    costs_clean = [compute_cost(y, t) for t in theta1_values]
    theta1_min_clean = theta1_values[np.argmin(costs_clean)]
    return theta1_values, costs_clean, theta1_min_clean

# с шумом
def analyze_noisy_data(x, y, noise_seed=42):
    np.random.seed(noise_seed)
    noise = np.random.uniform(-2, 2, size=y.shape)
    y_noisy = y + noise
    theta1_values = np.linspace(0, 2, 100)
    costs_noisy = [compute_cost(y_noisy, t) for t in theta1_values]
    theta1_min_noisy = theta1_values[np.argmin(costs_noisy)]
    return theta1_values, costs_noisy, theta1_min_noisy, y_noisy

# данные
x = np.arange(1, 21)
y = x.copy()
m = len(x)

# анализ чистых данных
theta1_values, costs_clean, theta1_min_clean = analyze_clean_data(x, y)

# анализ зашумленных данных
theta1_values, costs_noisy, theta1_min_noisy, y_noisy = analyze_noisy_data(x, y)

# Визуализация
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(x, y, 'o', label='Чистые данные')
plt.plot(x, y_noisy, 'o', label='Шумные данные')
plt.plot(x, h(x, 1.0), label='θ₁ = 1.0')
plt.plot(x, h(x, theta1_min_clean), 'g--', label=f'θ₁ min (чист) = {theta1_min_clean:.2f}')
plt.plot(x, h(x, theta1_min_noisy), 'r--', label=f'θ₁ min (шум) = {theta1_min_noisy:.2f}')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Аппроксимация данных прямыми")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(theta1_values, costs_clean, label='J(θ₁) — без шума')
plt.plot(theta1_values, costs_noisy, label='J(θ₁) — с шумом')
plt.axvline(theta1_min_clean, color='g', linestyle='--')
plt.axvline(theta1_min_noisy, color='r', linestyle='--')
plt.xlabel("θ₁")
plt.ylabel("J(θ₁)")
plt.title("График функции ошибки")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()