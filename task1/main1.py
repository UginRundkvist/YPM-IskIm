import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# 1) Символический расчёт объёма
# -----------------------------
y = sp.symbols('y', nonnegative=True)
V = sp.integrate(4*sp.sqrt(y)*(4 - y), (y, 0, 4))
print("Объём =", sp.simplify(V))  # 512/15

# -----------------------------
# 2) Визуализация области
# -----------------------------
# Область D: 0 <= y <= 4, 0 <= x <= 4 - y
ny, nx = 120, 120
Y = np.linspace(0.0, 4.0, ny)
X_list, Y_list = [], []
for yv in Y:
    xmax = 4.0 - yv
    if xmax <= 0:
        continue
    X = np.linspace(0.0, xmax, nx)
    Yrow = np.full_like(X, yv)
    X_list.append(X)
    Y_list.append(Yrow)

Xg = np.vstack(X_list)
Yg = np.vstack(Y_list)

# Верхняя поверхность z = 4*sqrt(y), нижняя z = 0
Z_top = 4*np.sqrt(Yg)
Z_bottom = np.zeros_like(Z_top)

# "Стенка" x + y = 4 (при y в [0,4], x = 4 - y), 0 <= z <= 4*sqrt(y)
Y_wall = np.linspace(0.0, 4.0, 200)
X_wall = 4.0 - Y_wall
Z_wall_top = 4*np.sqrt(Y_wall)

# Построение
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Верхняя поверхность
ax.plot_surface(Xg, Yg, Z_top, alpha=0.7, color='tab:orange', edgecolor='none', label='z = 4√y')

# Несколько "срезов" для заполнения объёма
kz = 10
for k in range(1, kz):
    t = k / kz
    Z_fill = (1 - t)*Z_bottom + t*Z_top
    ax.plot_surface(Xg, Yg, Z_fill, alpha=0.08, color='gray', edgecolor='none')

# Нижняя поверхность z=0 (контур)
ax.plot_surface(Xg, Yg, Z_bottom, alpha=0.2, color='tab:blue', edgecolor='none')

# Стенка x + y = 4
# Для каждой точки "стенки" проводим вертикальный отрезок от 0 до Z_wall_top
for i in range(len(Y_wall)):
    xw = X_wall[i]
    yw = Y_wall[i]
    zw = Z_wall_top[i]
    ax.plot([xw, xw], [yw, yw], [0, zw], color='k', alpha=0.5)

# Оформление
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Тело: 0<=z<=4√y, x>=0, y>=0, x+y<=4')
ax.view_init(elev=22, azim=-50)
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_zlim(0, 8)

plt.tight_layout()
plt.show()
