import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

MODEL_FILE = "trained_models.pkl"

def train_models():
    data = np.loadtxt("/home/zerd/all/YPM-IskIm/labe4/ex2data1.txt", delimiter=",")
    X_raw = data[:, :2]
    y = data[:, 2]

    x_min, x_max = X_raw.min(axis=0), X_raw.max(axis=0)
    X_norm = (X_raw - x_min) / (x_max - x_min)
    X = np.c_[np.ones(X_norm.shape[0]), X_norm]

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def hypothesis(X, theta):
        return sigmoid(np.dot(X, theta))

    def gradient_descent(X, y, theta, alpha, iters):
        m = len(y)
        for _ in range(iters):
            gradient = np.dot(X.T, (hypothesis(X, theta) - y)) / m
            theta -= alpha * gradient
        return theta

    # Линейная модель
    theta_linear = np.zeros(3)
    theta_linear = gradient_descent(X, y, theta_linear, alpha=0.01, iters=50000)

    # x1*x2 (гипербола)
    X_norm_hyper = np.column_stack([X_norm, X_norm[:, 0] * X_norm[:, 1]])
    X_hyper = np.c_[np.ones(X_norm_hyper.shape[0]), X_norm_hyper]
    theta_hyper = np.zeros(4)
    theta_hyper = gradient_descent(X_hyper, y, theta_hyper, alpha=0.1, iters=50000)

    #  параболо (x1^2 и x1)
    X_norm_parabola = np.column_stack([X_norm, X_norm[:, 0]**2])
    X_parabola = np.c_[np.ones(X_norm_parabola.shape[0]), X_norm_parabola]
    theta_parabola = np.zeros(4)
    theta_parabola = gradient_descent(X_parabola, y, theta_parabola, alpha=0.1, iters=50000)

    #эллипс (x1^2 и x2^2)
    X_norm_ellipse = np.column_stack([X_norm, X_norm[:, 0]**2, X_norm[:, 1]**2])
    X_ellipse = np.c_[np.ones(X_norm_ellipse.shape[0]), X_norm_ellipse]
    theta_ellipse = np.zeros(5)
    theta_ellipse = gradient_descent(X_ellipse, y, theta_ellipse, alpha=0.1, iters=50000)
    
    models = {
        'theta_linear': theta_linear,
        'theta_hyper': theta_hyper,
        'theta_parabola': theta_parabola,
        'theta_ellipse': theta_ellipse,
        'x_min': x_min,
        'x_max': x_max,
        'X_raw': X_raw,
        'y': y
    }
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(models, f)
    
    return models

def load_models():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            models = pickle.load(f)
        return models
    else:
        return train_models()

models = load_models()

theta_linear = models['theta_linear']
theta_hyper = models['theta_hyper']
theta_parabola = models['theta_parabola']
theta_ellipse = models['theta_ellipse']
x_min = models['x_min']
x_max = models['x_max']
X_raw = models['X_raw']
y = models['y']

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

# Ввод данных пользователя
vib = float(input("Введите вибрацию трактора: "))
rot = float(input("Введите неравномерность вращения: "))
vib_norm = (vib - x_min[0]) / (x_max[0] - x_min[0])
rot_norm = (rot - x_min[1]) / (x_max[1] - x_min[1])

prob_linear = hypothesis(np.array([1.0, vib_norm, rot_norm]), theta_linear)
prob_hyper = hypothesis(np.array([1.0, vib_norm, rot_norm, vib_norm * rot_norm]), theta_hyper)
prob_parabola = hypothesis(np.array([1.0, vib_norm, rot_norm, vib_norm**2]), theta_parabola)
prob_ellipse = hypothesis(np.array([1.0, vib_norm, rot_norm, vib_norm**2, rot_norm**2]), theta_ellipse)

status_linear = "НЕИСПРАВЕН" if prob_linear >= 0.5 else "ИСПРАВЕН"
status_hyper = "НЕИСПРАВЕН" if prob_hyper >= 0.5 else "ИСПРАВЕН"
status_parabola = "НЕИСПРАВЕН" if prob_parabola >= 0.5 else "ИСПРАВЕН"
status_ellipse = "НЕИСПРАВЕН" if prob_ellipse >= 0.5 else "ИСПРАВЕН"

print(f"Линейная модель: {status_linear} (вероятность: {prob_linear:.3f})")
print(f"Гипербола (x1*x2): {status_hyper} (вероятность: {prob_hyper:.3f})")
print(f"Парабола (x1^2): {status_parabola} (вероятность: {prob_parabola:.3f})")
print(f"Эллипс (x1^2, x2^2): {status_ellipse} (вероятность: {prob_ellipse:.3f})")

#Визуализация
plt.figure(figsize=(12, 10))

plt.scatter(X_raw[y==0,0], X_raw[y==0,1], c='green', label='Исправен', alpha=0.6, edgecolors='black', s=50)
plt.scatter(X_raw[y==1,0], X_raw[y==1,1], c='red', label='Неисправен', alpha=0.6, edgecolors='black', s=50)

x_vals = np.linspace(X_raw[:,0].min() - 0.5, max(X_raw[:,0].max() + 0.5, 500), 100)
x_vals_norm = (x_vals - x_min[0])/(x_max[0]-x_min[0])
y_vals_norm = [-(theta_linear[0]+theta_linear[1]*x)/theta_linear[2] for x in x_vals_norm]
y_vals = x_min[1] + np.array(y_vals_norm)*(x_max[1]-x_min[1])
plt.plot(x_vals, y_vals, c='blue', linewidth=2, label='Линейная (прямая)', linestyle='--')

max_x = max(X_raw[:,0].max(), vib, 500) + 1
max_y = max(X_raw[:,1].max(), rot, 500) + 1
min_x = min(X_raw[:,0].min(), vib, 0) - 1
min_y = min(X_raw[:,1].min(), rot, 0) - 1

xx, yy = np.meshgrid(np.linspace(min_x, max_x, 200),
                     np.linspace(min_y, max_y, 200))
xx_norm = (xx - x_min[0])/(x_max[0]-x_min[0])
yy_norm = (yy - x_min[1])/(x_max[1]-x_min[1])

#x1*x2
Z_hyper = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x_test = np.array([1.0, xx_norm[i,j], yy_norm[i,j], xx_norm[i,j]*yy_norm[i,j]])
        Z_hyper[i,j] = hypothesis(x_test, theta_hyper)
plt.contour(xx, yy, Z_hyper, levels=[0.5], colors='orange', linewidths=2, linestyles='-')
plt.plot([], [], color='orange', linewidth=2, label='Гипербола (x1*x2)')

#(x1^2)
Z_parabola = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x_test = np.array([1.0, xx_norm[i,j], yy_norm[i,j], xx_norm[i,j]**2])
        Z_parabola[i,j] = hypothesis(x_test, theta_parabola)
plt.contour(xx, yy, Z_parabola, levels=[0.5], colors='red', linewidths=2, linestyles='--')
plt.plot([], [], color='red', linewidth=2, label='Парабола (x1^2)')

#x1^2 и x2^2
Z_ellipse = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x_test = np.array([1.0, xx_norm[i,j], yy_norm[i,j], xx_norm[i,j]**2, yy_norm[i,j]**2])
        Z_ellipse[i,j] = hypothesis(x_test, theta_ellipse)
plt.contour(xx, yy, Z_ellipse, levels=[0.5], colors='purple', linewidths=2, linestyles=':')
plt.plot([], [], color='purple', linewidth=2, label='Эллипс (x1^2, x2^2)')

# Точка пользователя
plt.scatter([vib], [rot], c='purple', s=50, marker='o', 
            label=f'Ваша точка', 
            edgecolors='black', linewidth=1, zorder=5)

plt.xlim(min_x, max(max_x, 500))
plt.ylim(min_y, max(max_y, 500))

plt.xlabel("Вибрация")
plt.ylabel("Неравномерность вращения")
plt.title("Сравнение моделей логистической регрессии")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()