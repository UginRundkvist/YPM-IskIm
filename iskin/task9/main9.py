import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

theta = np.array([1, 0, 2, 1, 1, 0])

def mat_model_output(x1, x2, theta):
    theta0, theta1, theta2, theta3, theta4, theta5 = theta
    return (theta0 +
            theta1 * x1 +
            theta2 * x2 +
            theta3 * x1 * x2 +
            theta4 * x1**2 +
            theta5 * x2**2)

a_values = [0.2, 0.5, 0.8]
def calculate_logit_c(a):
    return np.log(a / (1 - a))

colors = ['blue', 'black', 'red']
linestyles = ['--', '-', '-.']
C_values = [calculate_logit_c(a) for a in a_values]

#Сетка
x1_min, x1_max = -40, 20
x2_min, x2_max = -20, 40
resolution = 500
x1_grid = np.linspace(x1_min, x1_max, resolution)
x2_grid = np.linspace(x2_min, x2_max, resolution)
xx1, xx2 = np.meshgrid(x1_grid, x2_grid)

Z = mat_model_output(xx1, xx2, theta)

plt.figure(figsize=(10, 8))

base_cmap = plt.colormaps.get_cmap('coolwarm')
cmap_list = ['green', 'red']
custom_cmap = plt.matplotlib.colors.ListedColormap(cmap_list)

# Заливка
C_for_filling = calculate_logit_c(0.5)
classification_map = (Z > C_for_filling).astype(int)

plt.pcolormesh(xx1, xx2, classification_map, cmap=custom_cmap, shading='auto', alpha=0.4)

# 2. Построение границ для каждого C
for a_val, C_val, color, style in zip(a_values, C_values, colors, linestyles):
    plt.contour(xx1, xx2, Z, levels=[C_val], colors=color, linewidths=2, linestyles=style)

# асимптота
if x1_min <= -2 <= x1_max:
    plt.axvline(-2, color='gray', linestyle=':', linewidth=1)

# легенда
plt.title(f'Отображение гиперболической границы')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

legend_elements = [
    Rectangle((0, 0), 1, 1, fc=custom_cmap(1), alpha=0.4, label='Класс 1 ($f > 0$)'),
    Rectangle((0, 0), 1, 1, fc=custom_cmap(0), alpha=0.4, label='Класс 0 ($f < 0$)')
] + [
    Line2D([0], [0], color=c, lw=2, linestyle=s,
           label=f'Граница $a = {v}$ ($C={C_v:.2f}$)')
    for v, C_v, c, s in zip(a_values, C_values, colors, linestyles)
]

plt.legend(handles=legend_elements, loc='upper right', fontsize='small')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()