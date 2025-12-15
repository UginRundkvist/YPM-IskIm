import numpy as np
import matplotlib.pyplot as plt

#31 параметр

theta = np.array([
    1, 0, 2, 1, 1, 0,   
    0.1, -0.1, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.01,
    0.005, -0.005, 0.003, -0.003, 0.002, 0.002, 0.001, -0.001,
    0.0005, -0.0005, 0.0003, -0.0003, 0.0002, 0.0002, 0.0001, -0.0001
])



#theta = np.array([1 for i in range(31)])


# a
a_values = [0.5, 0.8, 0.2]

def calculate_K(a):
    if a <= 0 or a >= 1:
        return np.nan
    return np.log((1 - a) / a)

K_values = [calculate_K(a) for a in a_values]

# --- (30 слагаемых) ---
def polynomial_grid(x1, x2, theta):
    res = (theta[0] +
           theta[1] * x1 +
           theta[2] * x2 +
           theta[3] * x1 * x2 +
           theta[4] * x1**2 +
           theta[5] * x2**2)

    res += theta[6] * x1**3
    res += theta[7] * x2**3
    res += theta[8] * (x1**2) * x2
    res += theta[9] * x1 * (x2**2)
    res += theta[10] * x1**4
    res += theta[11] * x2**4
    res += theta[12] * (x1**3) * x2
    res += theta[13] * x1 * (x2**3)
    res += theta[14] * (x1**2) * (x2**2)
    res += theta[15] * x1**5
    res += theta[16] * x2**5
    res += theta[17] * (x1**4) * x2
    res += theta[18] * x1 * (x2**4)
    res += theta[19] * (x1**3) * (x2**2)
    res += theta[20] * (x1**2) * (x2**3)
    res += theta[21] * x1**6
    res += theta[22] * x2**6
    res += theta[23] * (x1**5) * x2
    res += theta[24] * x1 * (x2**5)
    res += theta[25] * (x1**4) * (x2**2)
    res += theta[26] * (x1**2) * (x2**4)
    res += theta[27] * (x1**3) * (x2**3)
    res += theta[28] * x1**7
    res += theta[29] * x2**7
    res += theta[30] * (x1**6) * x2

    return res

# --- Решение уравнения по x2 ---
def find_roots_for_x1(x1_val, K, theta):
    # коэффициенты для многочлена по x2
    c7 = theta[29]
    c6 = theta[22]
    c5 = theta[16] + theta[24] * x1_val
    c4 = theta[11] + theta[18] * x1_val + theta[26] * (x1_val**2)
    c3 = theta[7] + theta[13] * x1_val + theta[20] * (x1_val**2) + theta[27] * (x1_val**3)
    c2 = theta[5] + theta[9] * x1_val + theta[14] * (x1_val**2) + theta[19] * (x1_val**3) + theta[25] * (x1_val**4)
    c1 = (theta[2] +
          theta[3] * x1_val +
          theta[8] * (x1_val**2) +
          theta[12] * (x1_val**3) +
          theta[17] * (x1_val**4) +
          theta[23] * (x1_val**5))
    c0 = (theta[0] +
          theta[1] * x1_val +
          theta[4] * (x1_val**2) +
          theta[6] * (x1_val**3) +
          theta[10] * (x1_val**4) +
          theta[15] * (x1_val**5) +
          theta[21] * (x1_val**6) +
          theta[28] * (x1_val**7) +
          theta[30] * (x1_val**6) * x1_val - K)

    coeffs = [c7, c6, c5, c4, c3, c2, c1, c0]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    return real_roots

# точки по x1
lines_data = {}
x_min, x_max = -5.0, 5.0
y_min, y_max = -8.0, 8.0
x_scan = np.linspace(x_min, x_max, 5000)

for a, K in zip(a_values, K_values):
    x_points, y_points = [], []
    for x_val in x_scan:
        roots = find_roots_for_x1(x_val, K, theta)
        for r in roots:
            if y_min <= r <= y_max:
                x_points.append(x_val)
                y_points.append(r)
    lines_data[a] = (x_points, y_points)

# --- Визуализация ---
plt.figure(figsize=(10, 8))
plt.title(r'Линии уровня и Области Классификации', fontsize=14)
plt.xlabel('x_1', fontsize=12)
plt.ylabel('x_2', fontsize=12)

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))

Z_raw = polynomial_grid(xx, yy, theta)
Z_class = np.where(Z_raw >= 0, 1, 0)

colors_fill = ['#dceeff', '#ffdcdc']
plt.pcolormesh(xx, yy, Z_class,
               cmap=plt.matplotlib.colors.ListedColormap(colors_fill),
               shading='auto', alpha=0.6, zorder=1) #заливка

plt.plot([], [], color='#ffdcdc', linewidth=10, label=r'Область Класса 1 (P ≥ 0)', alpha=0.7)
plt.plot([], [], color='#dceeff', linewidth=10, label=r'Область Класса 0 (P < 0)', alpha=0.7)

colors = {0.5: 'black', 0.8: 'blue', 0.2: 'red'}
labels = {0.5: 'Граница (a=0.5)', 0.8: 'P=0.8 (a=0.8)', 0.2: 'P=0.2 (a=0.2)'}

for a in a_values:
    xp, yp = lines_data[a]
    plt.scatter(xp, yp, s=2, color=colors[a], label=labels[a], zorder=10)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

leg = plt.legend(loc='upper right', framealpha=0.9)
for lh in leg.legend_handles:
    if isinstance(lh, plt.matplotlib.lines.Line2D):
        lh.set_linewidth(2)
    else:
        lh._sizes = [30]

plt.grid(True, linestyle='--', alpha=0.4, zorder=0)
plt.show()
