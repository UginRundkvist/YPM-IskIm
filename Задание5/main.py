import numpy as np
import matplotlib.pyplot as plt

# Данные
x = np.arange(1, 21, dtype=float)     # x ∈ {1, ..., 20}
m = len(x)

y_clean = x.copy()                     # тренд без шума

# шум [-3, 3]
rng = np.random.default_rng(42)        # для воспроизводимости
noise = rng.uniform(-3, 3, size=m)
y_noisy = x + noise

# Стоимость J(θ1) при θ0=0: J = (1/(2m)) * Σ (θ1 * x_i - y_i)^2
def J_grid(theta1_grid, x, y):
    theta1_grid = np.asarray(theta1_grid)
    return (1.0 / (2 * len(x))) * np.sum((theta1_grid[:, None] * x - y) ** 2, axis=1)

def J_single(theta1, x, y):
    return (1.0 / (2 * len(x))) * np.sum((theta1 * x - y) ** 2)

# Аналитический минимум по θ1 при θ0=0: θ1* = (x^T y) / (x^T x)
theta1_star_clean = (x @ y_clean) / (x @ x)
theta1_star_noisy = (x @ y_noisy) / (x @ x)

# Диапазон для θ1
theta1_grid = np.linspace(-1.0, 3.0, 600)

# Значения функционала
J_clean = J_grid(theta1_grid, x, y_clean)
J_noisy = J_grid(theta1_grid, x, y_noisy)

# Минимальные значения
Jmin_clean = J_single(theta1_star_clean, x, y_clean)
Jmin_noisy = J_single(theta1_star_noisy, x, y_noisy)

# График
plt.figure(figsize=(8, 5))
plt.plot(theta1_grid, J_clean, label='J(θ1) для y = x', lw=2)
plt.plot(theta1_grid, J_noisy, label='J(θ1) для ỹ = x + шум', lw=2)

# Отметим минимумы
plt.axvline(theta1_star_clean, color='C0', ls='--', alpha=0.7)
plt.axvline(theta1_star_noisy, color='C1', ls='--', alpha=0.7)
plt.scatter([theta1_star_clean], [Jmin_clean], color='C0', zorder=3)
plt.scatter([theta1_star_noisy], [Jmin_noisy], color='C1', zorder=3)

plt.title('Функционал стоимости J(θ1) при θ0 = 0')
plt.xlabel('θ1')
plt.ylabel('J(θ1)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Для справки выведем найденные минимумы
print(f"Минимум без шума: θ1* = {theta1_star_clean:.4f}, J* = {Jmin_clean:.6f}")
print(f"Минимум с шумом:  θ1* = {theta1_star_noisy:.4f}, J* = {Jmin_noisy:.6f}")