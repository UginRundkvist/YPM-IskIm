import numpy as np
import matplotlib.pyplot as plt


# сигмоида
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная сигмоиды, выраженная через саму сигмоиду
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Производная гиперболического тангенса
def tanh_derivative(x):
    return 1 - np.tanh(x)**2

points_x = [0, 3, -3, 8, -8, 15, -15]

points_y = [sigmoid(x) for x in points_x]

print("Значения функции сигмоида")
for x, y in zip(points_x, points_y):
    print(f"x = {x:>3}: y = {y:,.10f}")


fig, ax = plt.subplots(1, 2, figsize=(16, 6))


# Диапазон x
x_sigmoid = np.linspace(-10, 10, 500)
y_sigmoid = sigmoid(x_sigmoid)

ax[0].plot(x_sigmoid, y_sigmoid, label=r'$y(x) = \frac{1}{1 + e^{-x}}$', color='blue')
# Отмечаем заданные точки
ax[0].scatter(points_x, points_y, color='red', marker='o', label='Вычисленные точки')

ax[0].set_title('График функции Сигмоида (Sigmoid)')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y(x)')
ax[0].grid(True, linestyle='--', alpha=0.6)
ax[0].legend()
ax[0].axhline(0.5, color='gray', linestyle=':')
ax[0].axhline(1.0, color='gray', linestyle=':')
ax[0].axhline(0.0, color='gray', linestyle=':')
ax[0].axvline(0.0, color='black', linewidth=0.5, linestyle='-')




x_hyp = np.linspace(-3, 3, 500)
x_hyp_no_zero = x_hyp[x_hyp != 0] # Массив без нуля

# Вычисляем значения
y_sinh = np.sinh(x_hyp)  
y_cosh = np.cosh(x_hyp)  
y_tanh = np.tanh(x_hyp)  
y_coth = 1.0 / np.tanh(x_hyp_no_zero)

ax[1].plot(x_hyp, y_cosh, label=r'$\cosh(x)$', color='red', linewidth=2)
ax[1].plot(x_hyp, y_sinh, label=r'$\sinh(x)$', color='green', linewidth=2)
ax[1].plot(x_hyp, y_tanh, label=r'$\tanh(x)$', color='blue', linewidth=2)
ax[1].plot(x_hyp_no_zero, y_coth, label=r'$\coth(x)$', color='brown', linewidth=2)


ax[1].set_title('Графики $\sinh(x), \cosh(x), \tanh(x), \coth(x)$')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

# оси координат
ax[1].axhline(0, color='black', linewidth=0.5, linestyle='-')
ax[1].axvline(0, color='black', linewidth=0.5, linestyle='-') # Вертикальная асимптота

# Асимптоты
ax[1].axhline(1, color='gray', linestyle=':', alpha=0.6)
ax[1].axhline(-1, color='gray', linestyle=':', alpha=0.6)

ax[1].set_ylim(-4, 4)
ax[1].grid(True, linestyle='--', alpha=0.6)
ax[1].legend()


plt.tight_layout()
plt.show()
