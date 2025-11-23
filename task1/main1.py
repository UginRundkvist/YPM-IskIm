import numpy as np
import matplotlib.pyplot as plt

a = 1.5               
mode = 'y_equals_x'     # 'y_equals_x' или 'y_const'
y_const = 0.0           # используется, если mode == 'y_const'
xlim = (-6.0, 6.0)      # диапазон x на графике
n_points = 6000         # плотность сетки
show_singularity_lines = True  # показывать вертикальные линии характеристик

def k_xy(X, Y, a):
    """
    Возвращает значения k(X,Y;a) с маскировкой недопустимых точек (NaN).
    X, Y — numpy-массивы одинаковой формы.
    """
    inside = a * X**2 - 0.82
    # аргумент логарифма
    arg = np.sqrt(inside)

    # знаменатель (может быть ±inf или 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = 1.0 + np.log(arg)

    # основная формула
    with np.errstate(over='ignore', invalid='ignore'):
        value = (np.cos(X)**2 + np.sin(Y)**2) / denom \
                + np.exp(np.sin(X**2)) \
                + np.cbrt(np.abs(np.sin(X)))

    # маска недопустимых/неопределённых точек
    bad = (inside <= 0) | (~np.isfinite(denom)) | (np.abs(denom) < 1e-12) | (~np.isfinite(value))
    value = np.where(bad, np.nan, value)
    return value

def main():
    # сетка по x
    x = np.linspace(xlim[0], xlim[1], int(n_points))
    if mode == 'y_equals_x':
        y = x.copy()
        title_suffix = f"при y = x, a = {a}"
    elif mode == 'y_const':
        y = np.full_like(x, y_const, dtype=float)
        title_suffix = f"при y = {y_const}, a = {a}"
    else:
        raise ValueError("mode должен быть 'y_equals_x' или 'y_const'")

    # вычисление k(x)
    kx = k_xy(x, y, a)

    # характерные точки
    x_dom = np.sqrt(0.82 / a)                         # границы области: |x| > x_dom
    x_sing = np.sqrt((np.exp(-2.0) + 0.82) / a)       # возможная особая точка знаменателя

    # информативный вывод в консоль
    print("Параметры графика:")
    print(f"  a = {a}")
    print(f"  режим = {mode} ({'y=x' if mode=='y_equals_x' else f'y={y_const}'})")
    print(f"  диапазон x = {xlim}, точек = {len(x)}")
    print("Характерные значения по области определения:")
    print(f"  |x| > sqrt(0.82/a) = {x_dom:.6g}")
    print(f"  возможная особая точка: x = ±sqrt(e^(-2)+0.82)/sqrt(a) = {x_sing:.6g}")

    # построение графика
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=120)
    ax.plot(x, kx, color='tab:blue', lw=1.6, label='k(x)')

    # вертикальные линии характеристик
    if show_singularity_lines:
        def draw_vline(xc, ls, label=None):
            if np.isfinite(xc) and (xlim[0] < xc < xlim[1]):
                ax.axvline(xc, color='gray', ls=ls, lw=1.0, alpha=0.9, label=label)

        draw_vline(+x_dom, '--', r'|x| = sqrt(0.82/a)')
        draw_vline(-x_dom, '--')
        draw_vline(+x_sing, ':',  r'denom = 0 (исключить)')
        draw_vline(-x_sing, ':')

    ax.set_xlim(*xlim)
    ax.set_xlabel('x')
    ax.set_ylabel('k(x)')
    ax.set_title(f'Обычный график функции k(x) {title_suffix}')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
