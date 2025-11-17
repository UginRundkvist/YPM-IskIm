import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Функция расчета поверхности
def calculate_surface(x_vals, y_vals):
    # Вычисляем компоненты функции
    component1 = 2 * np.exp(-((x_vals - 5) ** 2 + (y_vals + 0.2) ** 2) / 1.5)
    component2 = 0.8 * np.exp(-((2 * x_vals + 3) ** 2 + ((0.2 * y_vals) - 4) ** 2) / 5)
    component3 = -4 * np.exp(-((x_vals - 0.2) ** 2 + (y_vals + 2) ** 2) / 2)
    
    return component1 + component2 + component3

# Параметры области построения
x_range = (-5, 8)
y_range = (-10, 30)
grid_resolution = 200

# Создание координатной сетки
x_coords = np.linspace(x_range[0], x_range[1], grid_resolution)
y_coords = np.linspace(y_range[0], y_range[1], grid_resolution)
X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

# Расчет значений функции
Z_values = calculate_surface(X_grid, Y_grid)

# Построение графиков
figure = plt.figure(figsize=(16, 6))

# 3D визуализация
ax_3d = figure.add_subplot(1, 2, 1, projection='3d')
surface_plot = ax_3d.plot_surface(X_grid, Y_grid, Z_values, 
                                 cmap='plasma', alpha=0.85, 
                                 linewidth=0.2, antialiased=True)
ax_3d.set_xlabel('Ось X')
ax_3d.set_ylabel('Ось Y')
ax_3d.set_zlabel('f(x, y)')
ax_3d.set_title('3D визуализация функции')
plt.colorbar(surface_plot, ax=ax_3d, shrink=0.5, aspect=10)
ax_3d.view_init(35, 55)

# Контурный график
ax_contour = figure.add_subplot(1, 2, 2)
contour_lines = ax_contour.contour(X_grid, Y_grid, Z_values, 
                                  levels=25, colors='darkblue', 
                                  linewidths=0.6)
filled_contours = ax_contour.contourf(X_grid, Y_grid, Z_values, 
                                     levels=60, cmap='plasma')
ax_contour.set_xlabel('Ось X')
ax_contour.set_ylabel('Ось Y')
ax_contour.set_title('Контурная диаграмма')
plt.colorbar(filled_contours, ax=ax_contour, shrink=0.5, aspect=1)
ax_contour.clabel(contour_lines, inline=True, fontsize=7, fmt='%.1f')

plt.tight_layout()
plt.show()

# Детальный контурный график
plt.figure(figsize=(10, 8))
detailed_contours = plt.contour(X_grid, Y_grid, Z_values, 
                               levels=35, colors='navy', 
                               linewidths=0.8)
detailed_filled = plt.contourf(X_grid, Y_grid, Z_values, 
                              levels=120, cmap='plasma')
color_bar = plt.colorbar(detailed_filled, shrink=0.8, aspect=12)
plt.clabel(detailed_contours, inline=True, fontsize=7, fmt='%.1f')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.title('Детальная контурная диаграмма')
plt.grid(True, alpha=0.25, linestyle='--')
plt.show()
