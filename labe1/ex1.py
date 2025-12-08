# import argparse
# import sys
# import os
import numpy as np
import matplotlib.pyplot as plt


def warmUpExercise(n: int): #создает единичную матрицу 2 способами!!
    # Способ 1: стандартный
    I_builtin = np.eye(n, dtype=float)

    # Способ 2: без стандартных функций
    I_manual = np.zeros((n, n), dtype=float)
    for i in range(n):
        I_manual[i, i] = 1.0
    return I_builtin, I_manual


# Создает первый график
def plotData(x, y, theta=None, title="Обучающяя выборка и регрессия"):#!!
    
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, c='blue', edgecolor='k', alpha=0.8, label='Данные')
    plt.xlabel('Количество автомобилей (x)')
    plt.ylabel('Прибыль СТО (y)')
    plt.title(title)
    if theta is not None and len(theta) == 2:
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = theta[0] + theta[1] * x_line #theta[0] - свободный член; theta[1] - коэффициент наклона
        plt.plot(x_line, y_line, 'r-', lw=2.0, label='h_θ(x) = θ0 + θ1 x')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#Делит данные на обучающую и тестовую части в пропорции 70/30.
def Train_test(x, y, test_size=0.3, shuffle=True, random_state=42):#!!
    assert len(x) == len(y)
    m = len(x)
    idx = np.arange(m)
    if shuffle: #перемешивает данные
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(idx)
    test_count = int(round(m * test_size))
    test_idx = idx[:test_count]
    train_idx = idx[test_count:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


# Преобразует в матрицу признаков X с первым столбцом из единиц.
def add_intercept(x):
    x = np.asarray(x).reshape(-1)
    m = x.shape[0]
    X = np.column_stack([np.ones(m), x])
    return X


#computeCost вычисляет функцию стоимости для одного параметра theta 

#через циклы
def computeCost_loop(X, y, theta):#!!
    m = len(y)
    cost_sum = 0.0
    for i in range(m):
        h_i = theta[0]*X[i,0] + theta[1]*X[i,1]
        diff = h_i - y[i]
        cost_sum += diff*diff
    return cost_sum / (2.0 * m)

# Поэлементная реализация без матричного умножения
def computeCost_elementwise(X, y, theta):
    h = theta[0]*X[:,0] + theta[1]*X[:,1]# Вектор предсказаний для всех примеров
    diff = h - y# Вектор предсказаний ошибок
    return np.sum(diff*diff) / (2.0 * len(y))

#Векторная реализация с матричными операциями.
def computeCost_vectorized(X, y, theta):
    diff = X @ theta - y
    return (diff @ diff) / (2.0 * len(y))


# Градиентный спуск (три реализации)

#Градиентный спуск с явными циклами..
def gradientDescent_loop(X, y, theta, alpha=1e-3, num_iters=1500):
    m = len(y)
    theta = theta.astype(float).copy()
    J_history = np.zeros(num_iters)
    for it in range(num_iters):
        # Считаем градиент через суммы
        grad0 = 0.0
        grad1 = 0.0
        for i in range(m):
            h_i = theta[0]*X[i,0] + theta[1]*X[i,1]
            err = h_i - y[i]
            grad0 += err * X[i,0]
            grad1 += err * X[i,1]
        grad0 /= m # усреднение
        grad1 /= m
        # обновление
        theta0_new = theta[0] - alpha * grad0
        theta1_new = theta[1] - alpha * grad1
        theta[0], theta[1] = theta0_new, theta1_new

        # Стоимость
        J_history[it] = computeCost_vectorized(X, y, theta)
    return theta, J_history

#Поэлементный градиентный спуск.
def gradientDescent_elementwise(X, y, theta, alpha=1e-3, num_iters=1500):
    m = len(y)
    theta = theta.astype(float).copy()
    J_history = np.zeros(num_iters)
    for it in range(num_iters):
        h = theta[0]*X[:,0] + theta[1]*X[:,1]
        err = h - y
        grad0 = np.sum(err * X[:,0]) / m
        grad1 = np.sum(err * X[:,1]) / m
        theta[0] -= alpha * grad0
        theta[1] -= alpha * grad1
        J_history[it] = np.sum((err)**2) / (2.0 * m)
    return theta, J_history

#Векторный градиентный спуск (быстрый).
def gradientDescent_vectorized(X, y, theta, alpha=1e-3, num_iters=1500):
    m = len(y)
    theta = theta.astype(float).copy()
    J_history = np.zeros(num_iters)
    for it in range(num_iters):
        err = X @ theta - y
        grad = (X.T @ err) / m
        theta -= alpha * grad
        J_history[it] = (err @ err) / (2.0 * m)
    return theta, J_history


# #Возвращает предсказание для скаляра или массива x с использованием theta
def predict(x, theta):
    x = np.asarray(x).reshape(-1)
    return theta[0] + theta[1] * x

#читает файл и загружает данные
def load_data(path):
    data = np.loadtxt(path, delimiter=',')
    return data[:, 0], data[:, 1]

def main():
    # Создание единичной матрицы
    n = int(input("Введите размер матрицы n: "))
    identity_matrix = np.eye(n)
    print("Единичная матрица:")
    print(identity_matrix)
    
    # Загрузка данных 
    x, y = load_data("/home/zerd/all/YPM-IskIm/labe1/lab1date.txt")
    x_train, y_train, x_test, y_test = Train_test(x, y, test_size=0.3)
    X_train = add_intercept(x_train)
    
    # Обучение модели
    theta, J_hist = gradientDescent_vectorized(X_train, y_train, np.array([0, 0]))
    
    theta_path = "Theta.txt"
    np.savetxt(theta_path, theta.reshape(1, -1), fmt="%.10f", header="theta0 theta1", comments="")
    print(f"Параметры модели сохранены в файл: {theta_path}")
    print(f"θ0 = {theta[0]:.6f}, θ1 = {theta[1]:.6f}")
    
    plotData(x_train, y_train, theta=theta)
    plt.show()

#указание на основную программу
if __name__ == "__main__":
    main()