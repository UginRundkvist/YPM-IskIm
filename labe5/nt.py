import numpy as np
import matplotlib.pyplot as plt

N = 21  

def features_variant_2_fast(x):
    m = x.shape[0]
    x1 = x[:, 1:2]
    x2 = x[:, 2:3]
    
    x1_powers = [np.ones((m, 1))]
    x2_powers = [np.ones((m, 1))]
    
    for d in range(1, N + 1):
        x1_powers.append(x1_powers[-1] * x1)  
        x2_powers.append(x2_powers[-1] * x2)
    
    features = [x[:, 0:1]]  
    for i in range(N + 1):
        for j in range(N + 1):
            if i == 0 and j == 0:
                continue
            features.append(x1_powers[i] * x2_powers[j])
    
    return np.hstack(features)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def cost_gradient(theta, features, y, lambda_1, lambda_2):
    m = features.shape[0]
    h = sigmoid(np.dot(features, theta))
    grad = (1/m) * np.dot(features.T, h - y)
    grad[1:] += (2 * lambda_1 / m) * theta[1:] + (lambda_2 / m) * np.sign(theta[1:])
    return grad

def train(x, y, iterations, alpha, lambda_1, lambda_2):
    features = features_variant_2_fast(x)
    
    nu = np.zeros((features.shape[1], 1))
    sigma = np.ones((features.shape[1], 1))
    
    m = features.shape[0]
    nu[1:] = np.mean(features, axis=0)[1:, None]
    diff = features - nu.T
    sigma[1:] = np.std(features, axis=0)[1:, None]
    sigma[sigma == 0] = 1
    features = diff / sigma.T
    
    theta = np.zeros((features.shape[1], 1))
    for _ in range(iterations):
        theta -= alpha * cost_gradient(theta, features, y, lambda_1, lambda_2)
    
    return theta, nu, sigma

def predict(x, theta, nu, sigma):
    features = features_variant_2_fast(x)
    features = (features - nu.T) / sigma.T
    return sigmoid(np.dot(features, theta))

def plot_decision_boundary(x, y, theta, nu, sigma, title, ax):
    faulty = y[:, 0] > 0.5
    working = y[:, 0] <= 0.5
    
    ax.scatter(x[faulty, 1], x[faulty, 2], c="red", marker="o", label="Неисправен", s=30)
    ax.scatter(x[working, 1], x[working, 2], c="green", marker="o", label="Исправен", s=30)
    
    x1_linspace = np.linspace(-4, 4, 150)
    x2_linspace = np.linspace(-4, 4, 150)
    x1_space, x2_space = np.meshgrid(x1_linspace, x2_linspace)
    
    X_grid = np.column_stack([
        np.ones(x1_space.size),
        x1_space.ravel(),
        x2_space.ravel()
    ])
    Z_grid = predict(X_grid, theta, nu, sigma)
    z_space = Z_grid.reshape(x1_space.shape)
    
    ax.contour(x1_space, x2_space, z_space, levels=[0.5], colors="black", linewidths=2)
    ax.set_xlabel("Вибрация")
    ax.set_ylabel("Неравномерность вращения")
    ax.set_title(title)
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best")

with open("/home/zerd/all/YPM-IskIm/labe5/data.txt", "r") as file:
    data = np.array([line.strip().split(",") for line in file], dtype=float)

x = np.hstack([np.ones((data.shape[0], 1)), data[:, :2]])
y = data[:, 2:3]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

print("Обучение без регуляризации...")
theta_no_reg, nu_no_reg, sigma_no_reg = train(x, y, 5000, 0.01, 0.0, 0.0)
plot_decision_boundary(x, y, theta_no_reg, nu_no_reg, sigma_no_reg, 
                      "Без регуляризации (λ₁=0, λ₂=0)", axes[0, 0])

print("Обучение с L2 регуляризацией...")
theta_l2, nu_l2, sigma_l2 = train(x, y, 5000, 0.01, 0.1, 0.0)
plot_decision_boundary(x, y, theta_l2, nu_l2, sigma_l2, 
                      "L2 регуляризация (λ₁=0.1, λ₂=0)", axes[0, 1])

print("Обучение с L1 регуляризацией...")
theta_l1, nu_l1, sigma_l1 = train(x, y, 5000, 0.01, 0.0, 0.05)
plot_decision_boundary(x, y, theta_l1, nu_l1, sigma_l1, 
                      "L1 регуляризация (λ₁=0, λ₂=0.05)", axes[1, 0])

print("Обучение с комбинированной регуляризацией...")
theta_elastic, nu_elastic, sigma_elastic = train(x, y, 5000, 0.01, 0.1, 0.05)
plot_decision_boundary(x, y, theta_elastic, nu_elastic, sigma_elastic, 
                      "Elastic Net (λ₁=0.1, λ₂=0.05)", axes[1, 1])

plt.tight_layout()
plt.show()