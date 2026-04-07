import numpy as np
import matplotlib.pyplot as plt

N = 5

def features_variant_2(x):
    x = np.array(x)  # Фикс: преобразуем в numpy-массив
    features = x[:, 0:1]
    x1 = x[:, 1:2]
    x2 = x[:, 2:3]
    for degree_1 in range(0, N + 1):
        for degree_2 in range(0, N + 1):
            features = np.hstack([features, (x1 ** degree_1) * (x2 ** degree_2)])
    return features

class classification():
    def __init__(self, features_count, features_map):
        self.theta = np.zeros((features_count, 1))
        self.nu = np.zeros((features_count, 1))
        self.sigma = np.ones((features_count, 1))
        self.features_map = features_map
    
    def hypothesis(self, features):
        z = np.dot(features, self.theta)
        z = np.clip(z, -500, 500)  # Фикс: защита от переполнения
        return 1 / (1 + np.exp(-z))
    
    def cost_gradient(self, features, y, lambda1, lambda2):
        m = features.shape[0]
        grad = (1/m) * np.dot(features.T, self.hypothesis(features) - y)
        grad[1:] += 2 * lambda1 * self.theta[1:] + lambda2 * np.sign(self.theta[1:])
        return grad
    
    def train(self, x, y, iterations, alpha, lambda1, lambda2):
        features = self.features_map(x)
        m = features.shape[0]
        self.nu[1:] = np.mean(features, axis=0)[1:, None]
        self.sigma[1:] = np.std(features, axis=0)[1:, None]
        self.sigma[self.sigma == 0] = 1  # Фикс: защита от деления на ноль
        features = (features - self.nu.T) / self.sigma.T
        
        for _ in range(iterations):
            self.theta -= alpha * self.cost_gradient(features, y, lambda1, lambda2)
    
    def predict(self, x):
        features = self.features_map(x)
        features = (features - self.nu.T) / self.sigma.T
        return self.hypothesis(features)

# Загрузка данных
with open("C:/Users/1/Desktop/IskIn/YPM-IskIm/labe5/data.txt", "r") as file:
    data = np.array([line.strip().split(",") for line in file], dtype=float)

x = np.hstack([np.ones((data.shape[0], 1)), data[:, :2]])
y = data[:, 2:3]

# Эксперименты с регуляризацией
configs = [
    ("Без регуляризации", 0, 0),
    ("L2 (Ridge)", 0.01, 0),
    ("L1 (Lasso)", 0, 0.01),
    ("L1+L2 (Elastic Net)", 0.01, 0.01),
    ("Сильная L2", 0.1, 0)
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (title, l1, l2) in enumerate(configs):
    print(f"Обучение: {title}...")
    model = classification((N+1)**2 + 1, features_variant_2)
    model.train(x, y, 2000, 0.01, l1, l2)
    
    # Рисуем точки
    for i in range(x.shape[0]):
        color = 'red' if y[i,0] > 0.5 else 'green'
        axes[idx].scatter(x[i,1], x[i,2], c=color, marker='x')
    
    # Рисуем границу
    xx, yy = np.meshgrid(np.linspace(-4, 4, 80), np.linspace(-4, 4, 80))
    zz = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = model.predict([[1, xx[i, j], yy[i, j]]])[0, 0]
    
    axes[idx].contour(xx, yy, zz, levels=[0.5], colors='black')
    axes[idx].set_title(f"{title}\nλ₁={l1}, λ₂={l2}")
    axes[idx].set_xlabel("Вибрация")
    axes[idx].set_ylabel("Вращение")
    axes[idx].grid(True, alpha=0.3)

axes[5].axis('off')
plt.tight_layout()
plt.show()

# Сравнение количества ненулевых весов
print("\n" + "="*50)
print("ВЛИЯНИЕ РЕГУЛЯРИЗАЦИИ НА ВЕСА МОДЕЛИ")
print("="*50)

for title, l1, l2 in configs:
    model = classification((N+1)**2 + 1, features_variant_2)
    model.train(x, y, 2000, 0.01, l1, l2)
    nonzero = np.sum(np.abs(model.theta[1:]) > 1e-5)
    print(f"{title:20} | Ненулевых весов: {nonzero:3} из {len(model.theta)-1}")