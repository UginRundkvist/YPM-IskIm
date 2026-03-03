import numpy as np
import matplotlib.pyplot as plt

# Чтение и нормализация данных
data = np.loadtxt("ex2data1.txt", delimiter=",")
X_raw = data[:, :2]
y = data[:, 2]

x_min, x_max = X_raw.min(axis=0), X_raw.max(axis=0)
X_norm = (X_raw - x_min) / (x_max - x_min)
X = np.c_[np.ones(X_norm.shape[0]), X_norm]

# Сигмоида
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

# Градиентный спуск
def gradient_descent(X, y, theta, alpha, iters):
    m = len(y)
    for _ in range(iters):
        gradient = np.dot(X.T, (hypothesis(X, theta) - y)) / m
        theta -= alpha * gradient
    return theta

# Линейная модель
theta_linear = np.zeros(3)
theta_linear = gradient_descent(X, y, theta_linear, alpha=0.01, iters=50000)

# Нелинейная модель с x1*x2
X_norm_nonlinear = np.column_stack([X_norm, X_norm[:, 0] * X_norm[:, 1]])
X_nonlinear = np.c_[np.ones(X_norm_nonlinear.shape[0]), X_norm_nonlinear]
theta_nonlinear = np.zeros(4)
theta_nonlinear = gradient_descent(X_nonlinear, y, theta_nonlinear, alpha=0.1, iters=50000)

# Ввод данных
vib = float(input("Введите вибрацию трактора: "))
rot = float(input("Введите неравномерность вращения: "))

vib_norm = (vib - x_min[0]) / (x_max[0] - x_min[0])
rot_norm = (rot - x_min[1]) / (x_max[1] - x_min[1])

# Предсказания
prob_linear = hypothesis(np.array([1.0, vib_norm, rot_norm]), theta_linear)
prob_nonlinear = hypothesis(np.array([1.0, vib_norm, rot_norm, vib_norm * rot_norm]), theta_nonlinear)

# Определяем состояние для каждой модели
status_linear = "НЕИСПРАВЕН" if prob_linear >= 0.5 else "ИСПРАВЕН"
status_nonlinear = "НЕИСПРАВЕН" if prob_nonlinear >= 0.5 else "ИСПРАВЕН"

print(f"\nВероятность неисправности:")
print(f"Линейная модель: {prob_linear:.3f} - Трактор {status_linear}")
print(f"Нелинейная модель: {prob_nonlinear:.3f} - Трактор {status_nonlinear}")

# Визуализация
plt.figure(figsize=(10, 8))

# Точки данных
plt.scatter(X_raw[y==0,0], X_raw[y==0,1], c='green', label='Исправен', alpha=0.6, edgecolors='black')
plt.scatter(X_raw[y==1,0], X_raw[y==1,1], c='red', label='Неисправен', alpha=0.6, edgecolors='black')

# Линейная граница
x_vals = np.linspace(X_raw[:,0].min(), X_raw[:,0].max(), 100)
x_vals_norm = (x_vals - x_min[0])/(x_max[0]-x_min[0])
y_vals_norm = [-(theta_linear[0]+theta_linear[1]*x)/theta_linear[2] for x in x_vals_norm]
y_vals = x_min[1] + np.array(y_vals_norm)*(x_max[1]-x_min[1])
plt.plot(x_vals, y_vals, c='blue', linewidth=2, label='Линейная', linestyle='--')

# Нелинейная граница
xx, yy = np.meshgrid(np.linspace(X_raw[:,0].min()-1, X_raw[:,0].max()+1, 100),
                     np.linspace(X_raw[:,1].min()-1, X_raw[:,1].max()+1, 100))
xx_norm = (xx - x_min[0])/(x_max[0]-x_min[0])
yy_norm = (yy - x_min[1])/(x_max[1]-x_min[1])

Z = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x_test = np.array([1.0, xx_norm[i,j], yy_norm[i,j], xx_norm[i,j]*yy_norm[i,j]])
        Z[i,j] = hypothesis(x_test, theta_nonlinear)

plt.contour(xx, yy, Z, levels=[0.5], colors='orange', linewidths=2)
plt.plot([], [], color='orange', linewidth=2, label='Нелинейная')

# Точка пользователя
plt.scatter([vib], [rot], c='purple', s=150, marker='o', 
            label=f'Ваша точка\n(лин:{prob_linear:.2f}, нелин:{prob_nonlinear:.2f})', 
            edgecolors='black', linewidth=2, zorder=5)

plt.xlabel("Вибрация")
plt.ylabel("Неравномерность вращения")
plt.title("Линейная vs Нелинейная логистическая регрессия")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()