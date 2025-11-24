import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ункция расчета поверхности представляет собой сумму трех гауссовых функций
def calculate_surface(x_vals, y_vals):
    
    # первый гауссов положительный пик
    component1 = 2 * np.exp(-((x_vals - 5) ** 2 + (y_vals + 0.2) ** 2) / 1.5)
    
    #Второй гауссов меньший положительный пик
    
    component2 = 0.8 * np.exp(-((2 * x_vals + 3) ** 2 + ((0.2 * y_vals) - 4) ** 2) / 5)
    
    # Третий гауссов отрицательный пик 
    component3 = -4 * np.exp(-((x_vals - 0.2) ** 2 + (y_vals + 2) ** 2) / 2)
    
    # Суммируем все компоненты для получения рельефа поверхности
    return component1 + component2 + component3


x_range = (-5, 8)  # Диапазон значений по оси X
y_range = (-10, 30)  # Диапазон значений по оси Y
grid_resolution = 200  #Kоличество точек в каждом направлении 

#создание координатной сетки для построения поверхности
# np.linspace создает равномерно распределенные точки в заданном диапазоне
x_coords = np.linspace(x_range[0], x_range[1], grid_resolution)
y_coords = np.linspace(y_range[0], y_range[1], grid_resolution)

# np.meshgrid создает матрицы координат из векторов x и y
X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

# Расчет значений функции на всей сетке координат
Z_values = calculate_surface(X_grid, Y_grid)

# Построение графиков 
figure = plt.figure(figsize=(16, 6)) 

# Создаем 3D подграфик 
ax_3d = figure.add_subplot(1, 2, 1, projection='3d')

# Построение поверхности в 3D
surface_plot = ax_3d.plot_surface(X_grid, Y_grid, Z_values, 
                                 cmap='plasma',  
                                 alpha=0.85,     
                                 linewidth=0.2, 
                                 antialiased=True) 

#Настройка осей и заголовка
ax_3d.set_xlabel('Ось X')
ax_3d.set_ylabel('Ось Y')
ax_3d.set_zlabel('f(x, y)')
ax_3d.set_title('3D визуализация функции')

#Добавление цветовой шкалы для отображения соответствия цветов значениям Z
plt.colorbar(surface_plot, ax=ax_3d, shrink=0.5, aspect=10)


# Создаем 2D подграфик для контурной диаграммы
ax_contour = figure.add_subplot(1, 2, 2)

# Рисуем контурные линии - изолинии (линии равной высоты)
contour_lines = ax_contour.contour(X_grid, Y_grid, Z_values, 
                                  levels=25,        # Количество уровней изолиний
                                  colors='darkblue', # Цвет линий
                                  linewidths=0.6)   # Толщина линий

# Рисуем заполненные контуры (цветные области между изолиниями)
filled_contours = ax_contour.contourf(X_grid, Y_grid, Z_values, 
                                     levels=60,     # Более детальная градация цветов
                                     cmap='plasma') # Та же цветовая карта

# Настройка осей и заголовка
ax_contour.set_xlabel('Ось X')
ax_contour.set_ylabel('Ось Y')
ax_contour.set_title('Контурная диаграмма')

# Добавление цветовой шкалы
plt.colorbar(filled_contours, ax=ax_contour, shrink=0.5, aspect=1)

# Добавление подписей значений на контурные линии
ax_contour.clabel(contour_lines, inline=True, fontsize=7, fmt='%.1f')

# Автоматическая регулировка расстояний между подграфиками
plt.tight_layout()
plt.show()