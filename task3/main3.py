import numpy as np

a = np.array([[3], [2], [1]])
b = np.array([[5], [6], [7]])

print("Вектор a:\n", a)
print("Вектор a:\n", b)

print("Размерность a:", a.shape)
print("Размерность b:", b.shape)

loop = 0
for i in range(3):
    loop += a[i][0] * b[i][0]
print("Скалярное произведение (цикл):", loop)

elementwise = np.sum(a * b)
print("Скалярное произведение (поэлементно):", elementwise)

matrix = np.dot(a.T,b)
print("Скалярное произведение (матрица):", matrix)
