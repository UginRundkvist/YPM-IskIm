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

def calculate_accuracy(x, y, theta, nu, sigma):
    predictions = predict(x, theta, nu, sigma)
    predictions_binary = (predictions >= 0.5).astype(int)
    accuracy = np.mean(predictions_binary == y) * 100
    return accuracy

def plot_decision_boundary(x, y, theta, nu, sigma, title, ax, acc):
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
    ax.set_title(f"{title}\nAccuracy: {acc:.2f}%", fontsize=11)
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best")

with open("C:/Users/1/Desktop/IskIn/YPM-IskIm/labe5/data.txt", "r") as file:
    data = np.array([line.strip().split(",") for line in file], dtype=float)

x = np.hstack([np.ones((data.shape[0], 1)), data[:, :2]])
y = data[:, 2:3]

np.random.seed(42)
indices = np.random.permutation(x.shape[0])
split = int(0.7 * x.shape[0])

train_idx = indices[:split]
test_idx = indices[split:]

x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

fig, axes = plt.subplots(3, 3, figsize=(16, 14))

# Без регуляризации (базовый, будет переобучение)
theta_no_reg, nu_no_reg, sigma_no_reg = train(x_train, y_train, 5000, 0.01, 0.0, 0.0)
train_acc_no_reg = calculate_accuracy(x_train, y_train, theta_no_reg, nu_no_reg, sigma_no_reg)
test_acc_no_reg = calculate_accuracy(x_test, y_test, theta_no_reg, nu_no_reg, sigma_no_reg)
print(f"Без регуляризации: train={train_acc_no_reg:.2f}%, test={test_acc_no_reg:.2f}%")

#L2
theta_l2, nu_l2, sigma_l2 = train(x_train, y_train, 5000, 0.01, 1.0, 0.0)
test_acc_l2 = calculate_accuracy(x_test, y_test, theta_l2, nu_l2, sigma_l2)
train_acc_l2 = calculate_accuracy(x_train, y_train, theta_l2, nu_l2, sigma_l2)
print(f"L2 (Ridge, λ=1.0): train={train_acc_l2:.2f}%, test={test_acc_l2:.2f}%")

# L1 
theta_l1, nu_l1, sigma_l1 = train(x_train, y_train, 5000, 0.01, 0.0, 0.5)
test_acc_l1 = calculate_accuracy(x_test, y_test, theta_l1, nu_l1, sigma_l1)
train_acc_l1 = calculate_accuracy(x_train, y_train, theta_l1, nu_l1, sigma_l1)
print(f"L1 (Lasso, λ=0.5): train={train_acc_l1:.2f}%, test={test_acc_l1:.2f}%")

#Elastic Net 
theta_elastic, nu_elastic, sigma_elastic = train(x_train, y_train, 5000, 0.01, 1.0, 0.5)
test_acc_elastic = calculate_accuracy(x_test, y_test, theta_elastic, nu_elastic, sigma_elastic)
train_acc_elastic = calculate_accuracy(x_train, y_train, theta_elastic, nu_elastic, sigma_elastic)
print(f"Elastic Net (L2=1.0, L1=0.5): train={train_acc_elastic:.2f}%, test={test_acc_elastic:.2f}%")

#Слишком слабая L2
theta_overfit_l2, nu_overfit_l2, sigma_overfit_l2 = train(x_train, y_train, 5000, 0.01, 0.0001, 0.0)
test_acc_overfit_l2 = calculate_accuracy(x_test, y_test, theta_overfit_l2, nu_overfit_l2, sigma_overfit_l2)
train_acc_overfit_l2 = calculate_accuracy(x_train, y_train, theta_overfit_l2, nu_overfit_l2, sigma_overfit_l2)
print(f"\nПЕРЕОБУЧЕНИЕ 1 (L2=0.0001): train={train_acc_overfit_l2:.2f}%, test={test_acc_overfit_l2:.2f}%")

#Слишком слабая L1
theta_overfit_l1, nu_overfit_l1, sigma_overfit_l1 = train(x_train, y_train, 5000, 0.01, 0.0, 0.0001)
test_acc_overfit_l1 = calculate_accuracy(x_test, y_test, theta_overfit_l1, nu_overfit_l1, sigma_overfit_l1)
train_acc_overfit_l1 = calculate_accuracy(x_train, y_train, theta_overfit_l1, nu_overfit_l1, sigma_overfit_l1)
print(f"ПЕРЕОБУЧЕНИЕ 2 (L1=0.0001): train={train_acc_overfit_l1:.2f}%, test={test_acc_overfit_l1:.2f}%")

# Оба лямбда почти нулевые (сильное переобучение) 
theta_overfit_both, nu_overfit_both, sigma_overfit_both = train(x_train, y_train, 5000, 0.01, 0.00001, 0.00001)
test_acc_overfit_both = calculate_accuracy(x_test, y_test, theta_overfit_both, nu_overfit_both, sigma_overfit_both)
train_acc_overfit_both = calculate_accuracy(x_train, y_train, theta_overfit_both, nu_overfit_both, sigma_overfit_both)
print(f"ПЕРЕОБУЧЕНИЕ 3 (L2=0.00001, L1=0.00001): train={train_acc_overfit_both:.2f}%, test={test_acc_overfit_both:.2f}%")

plot_decision_boundary(x, y, theta_no_reg, nu_no_reg, sigma_no_reg, 
                      "Без регуляризации (базовый)", axes[0, 0], test_acc_no_reg)

plot_decision_boundary(x, y, theta_l2, nu_l2, sigma_l2, 
                      "L2 регуляризация (λ=1.0)", axes[0, 1], test_acc_l2)

plot_decision_boundary(x, y, theta_l1, nu_l1, sigma_l1, 
                      "L1 регуляризация (λ=0.5)", axes[0, 2], test_acc_l1)

plot_decision_boundary(x, y, theta_elastic, nu_elastic, sigma_elastic, 
                      "Elastic Net (L2=1.0, L1=0.5)", axes[1, 0], test_acc_elastic)

plot_decision_boundary(x, y, theta_overfit_l2, nu_overfit_l2, sigma_overfit_l2, 
                      "ПЕРЕОБУЧЕНИЕ 1 (L2=0.0001)", axes[1, 1], test_acc_overfit_l2)

plot_decision_boundary(x, y, theta_overfit_l1, nu_overfit_l1, sigma_overfit_l1, 
                      "ПЕРЕОБУЧЕНИЕ 2 (L1=0.0001)", axes[1, 2], test_acc_overfit_l1)

plot_decision_boundary(x, y, theta_overfit_both, nu_overfit_both, sigma_overfit_both, 
                      "ПЕРЕОБУЧЕНИЕ 3 (очень слабые λ)", axes[2, 1], test_acc_overfit_both)

axes[2, 0].set_visible(False)
axes[2, 2].set_visible(False)

plt.tight_layout()
plt.show()