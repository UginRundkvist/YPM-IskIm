import numpy as np
import matplotlib.pyplot as plt

from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn
from computeCostMulti import computeCostMulti

# Загрузка данных
data = np.loadtxt(r"C:\Users\1\Desktop\IskIn\YPM-IskIm\labe2\ex1data2.txt", delimiter=',')

X = data[:, 0:2]  # Кол-во передач и скорость двигателя
y = data[:, 2]    # Стоимость трактора
m = len(y)

# Нормализация признаков
X_norm, mu, sigma = featureNormalize(X)

# Добавляем столбец единиц для градиентного спуска
X_norm_bias = np.c_[np.ones(m), X_norm]

# Начальные параметры для градиентного спуска
alpha = 0.01
num_iters = 400
theta = np.zeros(3)

# Градиентный спуск
theta_gd, J_history = gradientDescentMulti(X_norm_bias, y, theta, alpha, num_iters)
print("Theta (градиентный спуск):", theta_gd)

# График сходимости
plt.plot(range(num_iters), J_history)
plt.xlabel("Итерации")
plt.ylabel("J(theta)")
plt.title("Сходимость градиентного спуска")
plt.grid()
plt.show()

# Нормальное уравнение (аналитическое решение)
X_bias = np.c_[np.ones(m), X]
theta_normal = normalEqn(X_bias, y)
print("Theta (аналитическое решение):", theta_normal)

# --- Ввод данных пользователем ---
print("\nВведите данные нового трактора для предсказания стоимости:")
transmission = float(input("Скорость: "))
rpm = float(input("Количество передач: "))

# Подготовка данных для градиентного спуска
x_input_norm = np.array([transmission, rpm])
x_input_norm = (x_input_norm - mu) / sigma  # нормализация
x_input_norm = np.insert(x_input_norm, 0, 1)  # добавляем 1 для bias

# Подготовка данных для нормального уравнения
x_input = np.array([1, transmission, rpm])  # добавляем 1 для bias

# Предсказания
prediction_gd = x_input_norm @ theta_gd
prediction_normal = x_input @ theta_normal

# Вывод результатов
print(f"\nПредсказанная стоимость трактора (градиентный спуск): {prediction_gd:.2f}")
print(f"Предсказанная стоимость трактора (нормальное уравнение): {prediction_normal:.2f}")

