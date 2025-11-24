import numpy as np
import matplotlib.pyplot as plt

a = 1.5
x_range = (-6, 6)
num_points = 1000

def simple_k_xy(x, y, a):
 
    # подкоренное выражение
    inside = a * x**2 - 0.82
    valid_mask = inside > 0  #функция определена только здесь
    
    # знаменатель
    sqrt_inside = np.sqrt(inside)
    denominator = 1.0 + np.log(sqrt_inside)
    
    # показывает где знаменатель близок к нулю
    zero_denom_mask = np.abs(denominator) < 1e-12
    
    # вычисляем значение функции
    value = (np.cos(x)**2 + np.sin(y)**2) / denominator \
            + np.exp(np.sin(x**2)) \
            + np.cbrt(np.abs(np.sin(x)))
    
    # помечаем недопустимые точки как Nan
    value[~valid_mask] = np.nan
    value[zero_denom_mask] = np.nan
    
    return value

# Создаем точки для графика
x = np.linspace(x_range[0], x_range[1], num_points)
y = x 

kx = simple_k_xy(x, y, a)

# Вычисляем характерные точки
x_dom = np.sqrt(0.82 / a)
x_sing = np.sqrt((np.exp(-2.0) + 0.82) / a)

# строим график
plt.figure(figsize=(10, 6))
plt.plot(x, kx, 'b-', linewidth=1.5, label='k(x)')

# олтмечаем особые точки 
plt.axvline(x=x_dom, color='red', linestyle='--', alpha=0.7, label='Граница определения')
plt.axvline(x=-x_dom, color='red', linestyle='--', alpha=0.7)
plt.axvline(x=x_sing, color='orange', linestyle=':', alpha=0.7, label='Знаменатель = 0')
plt.axvline(x=-x_sing, color='orange', linestyle=':', alpha=0.7)

# Настраиваем график
plt.xlim(x_range)
plt.xlabel('x')
plt.ylabel('k(x)')
plt.title(f'График k(x) при y = x, a = {a}')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()