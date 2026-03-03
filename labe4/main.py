import math
import matplotlib.pyplot as plt
import numpy as np

# Чтение и нормализация данных
data = np.loadtxt("ex2data1.txt", delimiter=",")
X_raw = data[:, :2]
y = data[:, 2]

# нормализация
x_min, x_max = X_raw.min(axis=0), X_raw.max(axis=0)
X_norm = (X_raw - x_min) / (x_max - x_min)
X = np.c_[np.ones(X_norm.shape[0]), X_norm]  # добавляем θ0

#  Сигмоида 
def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))

def hypothesis(x, theta):
    return sigmoid(sum(theta[i]*x[i] for i in range(len(theta))))

# Градиентный спуск
def gradient_descent(X, y, theta, alpha, iters):
    m = len(y)
    for _ in range(iters):
        grad = [0]*len(theta)
        for i in range(m):
            h = hypothesis(X[i], theta)
            for j in range(len(theta)):
                grad[j] += (h - y[i])*X[i][j]
        for j in range(len(theta)):
            theta[j] -= alpha * grad[j]/m
    return theta

theta = [0.0, 0.0, 0.0]
theta = gradient_descent(X, y, theta, alpha=0.01, iters=50000)

# Предсказание 
vib = float(input("Введите вибрацию трактора: "))
rot = float(input("Введите неравномерность вращения: "))

x_new = [1.0, (vib-x_min[0])/(x_max[0]-x_min[0]), (rot-x_min[1])/(x_max[1]-x_min[1])]
prob = hypothesis(x_new, theta)
print(f"\nВероятность неисправности: {prob:.3f}")
print("Трактор НЕИСПРАВЕН" if prob>=0.5 else "Трактор ИСПРАВЕН")

# Визуализация 
plt.scatter(X_raw[y==0,0], X_raw[y==0,1], c='green', label='Исправен')
plt.scatter(X_raw[y==1,0], X_raw[y==1,1], c='red', label='Неисправен')

# линия разделения
x_vals = np.linspace(X_raw[:,0].min(), X_raw[:,0].max(), 100)
x_vals_norm = (x_vals - x_min[0])/(x_max[0]-x_min[0])
y_vals_norm = [-(theta[0]+theta[1]*x)/theta[2] for x in x_vals_norm]
y_vals = x_min[1] + np.array(y_vals_norm)*(x_max[1]-x_min[1])

plt.plot(x_vals, y_vals, c='blue', label='Граница 0.5')
plt.xlabel("Вибрация")
plt.ylabel("Неравномерность вращения")
plt.legend()
plt.title("Логистическая регрессия")
plt.show()