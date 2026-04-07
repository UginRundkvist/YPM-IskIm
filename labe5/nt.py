import numpy as np
import matplotlib.pyplot as plt

N = 5

def features_variant_2(x):
    features = x[:, 0:1]
    x1 = x[:, 1:2]
    x2 = x[:, 2:3]
    for degree_1 in range(0, N + 1):
        for degree_2 in range(0, N + 1):
            features = np.hstack([features, (x1 ** degree_1) * (x2 ** degree_2)])
    return features

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def cost_gradient(theta, features, y, lambda_1, lambda_2):
    m = features.shape[0]
    h = sigmoid(np.dot(features, theta))
    grad = (1/m) * np.dot(features.T, h - y)
    grad[1:] += 2 * lambda_1 * theta[1:] + lambda_2 * np.sign(theta[1:])
    return grad

def train(x, y, iterations, alpha, lambda_1, lambda_2):
    # Генерация признаков
    features = features_variant_2(x)
    
    # Нормализация
    nu = np.zeros((features.shape[1], 1))
    sigma = np.ones((features.shape[1], 1))
    
    m = features.shape[0]
    nu[1:] = np.mean(features, axis=0)[1:, None]
    diff = features - nu.T
    sigma[1:] = np.std(features, axis=0)[1:, None]
    sigma[sigma == 0] = 1
    features = diff / sigma.T
    
    # Обучение
    theta = np.zeros((features.shape[1], 1))
    for _ in range(iterations):
        theta -= alpha * cost_gradient(theta, features, y, lambda_1, lambda_2)
    
    return theta, nu, sigma

def predict(x, theta, nu, sigma):
    features = features_variant_2(x)
    features = (features - nu.T) / sigma.T
    z = np.dot(features, theta)
    return sigmoid(z)

# ============ ОСНОВНАЯ ЧАСТЬ ============
with open("C:/Users/1/Desktop/IskIn/YPM-IskIm/labe5/data.txt", "r") as file:
    data = np.array([line.strip().split(",") for line in file], dtype=float)

x = np.hstack([np.ones((data.shape[0], 1)), data[:, :2]])
y = data[:, 2:3]

theta, nu, sigma = train(x, y, 5000, 0.01, 0.0001, 0.0001)

# Визуализация
fig = plt.figure()
ax = fig.add_subplot()

for i in range(x.shape[0]):
    color = "red" if y[i, 0] > 0.5 else "green"
    ax.scatter(x[i, 1], x[i, 2], c=color, marker="x")

# Граница решения
x1_linspace = np.linspace(-4, 4, 150)
x2_linspace = np.linspace(-4, 4, 150)
x1_space, x2_space = np.meshgrid(x1_linspace, x2_linspace)
z_space = np.zeros_like(x1_space)

for i in range(x1_space.shape[0]):
    for j in range(x1_space.shape[1]):
        z_space[i, j] = predict(np.array([[1, x1_space[i, j], x2_space[i, j]]]), theta, nu, sigma)[0, 0]

ax.contour(x1_space, x2_space, z_space, levels=[0.5], colors="black")
ax.set_xlabel("Вибрация")
ax.set_ylabel("Неравномерность вращения")
ax.set_title(f"Модель с {N} признаками")
ax.grid(True, alpha=0.5)
ax.legend(["Неисправен", "Исправен"], loc="best")

plt.show()