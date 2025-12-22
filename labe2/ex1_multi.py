import numpy as np
import matplotlib.pyplot as plt

from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn

# Загрузка данных
data = np.loadtxt(r"C:\Users\1\Desktop\IskIn\YPM-IskIm\labe2\ex1data2.txt", delimiter=',')

X = data[:, 0:2]
y = data[:, 2]
m = len(y)

# Нормализация признаков
X_norm, mu, sigma = featureNormalize(X)

# Добавляем столбец единиц
X_norm = np.c_[np.ones(m), X_norm]

# Начальные параметры
alpha = 0.01
num_iters = 400
theta = np.zeros(3)

# Градиентный спуск
theta, J_history = gradientDescentMulti(
    X_norm, y, theta, alpha, num_iters
)

print("Theta (градиентный спуск):")
print(theta)

# График функции стоимости
plt.plot(range(num_iters), J_history)
plt.xlabel("Итерации")
plt.ylabel("J(theta)")
plt.title("Сходимость градиентного спуска")
plt.grid()
plt.show()

# Аналитическое решение
X = np.c_[np.ones(m), data[:, 0:2]]
theta_normal = normalEqn(X, y)

print("Theta (аналитическое решение):")
print(theta_normal)
