import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Исходные параметры модели
theta_base = np.array([1, 0, 2, 1, 1, 0])

# Генерируем 35 дополнительных нелинейных слагаемых с коэффициентами от -2 до 2
np.random.seed(42)
theta_extra = np.random.uniform(-2, 2, 35)

# Объединяем все коэффициенты
theta = np.concatenate([theta_base, theta_extra])

# Функция модели с 35 дополнительными слагаемыми
def mat_model_output_extended(x1, x2, theta):
    # первые 6 слагаемых
    val = theta[0] + theta[1]*x1 + theta[2]*x2 + theta[3]*x1*x2 + theta[4]*x1**2 + theta[5]*x2**2
    
    # дополнительные 35 слагаемых
    powers = [
        (3,0),(0,3),(2,1),(1,2),(4,0),(0,4),(3,1),(1,3),(2,2),(5,0),
        (0,5),(4,1),(1,4),(3,2),(2,3),(6,0),(0,6),(5,1),(1,5),(4,2),
        (2,4),(3,3),(7,0),(0,7),(6,1),(1,6),(5,2),(2,5),(4,3),(3,4),
        (2,5),(1,6),(0,7),(7,0),(6,1)
    ]
    for coef, (i,j) in zip(theta[6:], powers):
        val += coef * (x1**i) * (x2**j)
    return val

# Пороги
def calculate_logit_c(a):
    return np.log(a / (1 - a))

a_values = [0.2, 0.5, 0.8]
colors = ['blue', 'black', 'red']
linestyles = ['--', '-', '-.']
C_values = [calculate_logit_c(a) for a in a_values]

# Сетка
x1_min, x1_max = -5, 5
x2_min, x2_max = -5, 5
resolution = 500
x1_grid = np.linspace(x1_min, x1_max, resolution)
x2_grid = np.linspace(x2_min, x2_max, resolution)
xx1, xx2 = np.meshgrid(x1_grid, x2_grid)

# Вычисляем модель
Z = mat_model_output_extended(xx1, xx2, theta)

plt.figure(figsize=(10, 8))

# Заливка областей классов по порогу a=0.5
C_fill = calculate_logit_c(0.5)
classification_map = (Z > C_fill).astype(int)
plt.contourf(xx1, xx2, classification_map, levels=[-0.1,0.5,1.1], colors=['green','red'], alpha=0.4)

# Построение линий уровня для каждого порога
for a_val, C_val, color, style in zip(a_values, C_values, colors, linestyles):
    plt.contour(xx1, xx2, Z, levels=[C_val], colors=color, linewidths=2, linestyles=style)

# Настройки графика
plt.title('Расширенная модель с 35 нелинейными слагаемыми')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid(True, linestyle='--', alpha=0.6)

# Легенда
legend_elements = [
    Rectangle((0,0),1,1,fc='red', alpha=0.4, label='Класс 1 ($f > 0$)'),
    Rectangle((0,0),1,1,fc='green', alpha=0.4, label='Класс 0 ($f < 0$)')
] + [
    Line2D([0],[0], color=c, lw=2, linestyle=s, label=f'Граница $a={v}$ ($C={C_val:.2f}$)')
    for v, C_val, c, s in zip(a_values, C_values, colors, linestyles)
]

plt.legend(handles=legend_elements, loc='upper right', fontsize='small')
plt.show()
