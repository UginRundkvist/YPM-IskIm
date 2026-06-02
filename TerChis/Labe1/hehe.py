import math


def get_matrix_with_precision(sqrt_precision):
    s2 = round(math.sqrt(2), sqrt_precision)
    s3 = round(math.sqrt(3), sqrt_precision)

    A = [
        [4.0, 2.0 * s2, -2.0, 3.0 * s3],
        [1.0 + 1.0 / s2, 0.5 + 2.0 / s2, 0.0, -s3 / 2.0],
        [1 / 5, 3 / 5, 5 / 6, -1 / 2],
        [s2, 1.0, 1.0, 0.0]
    ]

    B = [0.18, 0.19, 0.21, 0.31]
    return A, B


def get_ideal_system():
    s2 = math.sqrt(2)
    s3 = math.sqrt(3)

    A = [
        [4.0, 2.0 * s2, -2.0, 3.0 * s3],
        [1.0 + 1.0 / s2, 0.5 + 2.0 / s2, 0.0, -s3 / 2.0],
        [1 / 5, 3 / 5, 5 / 6, -1 / 2],
        [s2, 1.0, 1.0, 0.0]
    ]
    B = [0.18, 0.19, 0.21, 0.31]
    return A, B


def copy_matrix(A, B):
    return [A[i][:] + [B[i]] for i in range(len(B))]


def back_substitution(M, n, col_order=None):
    x_temp = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(M[i][j] * x_temp[j] for j in range(i + 1, n))
        x_temp[i] = M[i][n] - sum_ax

    if col_order is None:
        return x_temp

    x_final = [0.0] * n
    for i, orig_idx in enumerate(col_order):
        x_final[orig_idx] = x_temp[i]
    return x_final


def solve_gauss_classic(A, B):
    n = len(B)
    M = copy_matrix(A, B)
    for i in range(n):
        pivot = M[i][i]
        if abs(pivot) < 1e-15:
            raise ZeroDivisionError("Pivot is zero")
        M[i] = [val / pivot for val in M[i]]
        for j in range(i + 1, n):
            factor = M[j][i]
            M[j] = [M[j][k] - factor * M[i][k] for k in range(n + 1)]
    return back_substitution(M, n)


def solve_gauss_pivot(A, B):
    n = len(B)
    M = copy_matrix(A, B)
    col_order = list(range(n))
    for i in range(n):
        max_val = -1.0
        pivot_col = i
        for col in range(i, n):
            if abs(M[i][col]) > max_val:
                max_val = abs(M[i][col])
                pivot_col = col

        if pivot_col != i:
            for row in range(n):
                M[row][i], M[row][pivot_col] = M[row][pivot_col], M[row][i]
            col_order[i], col_order[pivot_col] = col_order[pivot_col], col_order[i]

        pivot = M[i][i]
        if abs(pivot) < 1e-15:
            raise ZeroDivisionError("Pivot is zero")
        M[i] = [val / pivot for val in M[i]]
        for j in range(i + 1, n):
            factor = M[j][i]
            M[j] = [M[j][k] - factor * M[i][k] for k in range(n + 1)]
    return back_substitution(M, n, col_order)


def calculate_residual_vector(A_ideal, x, B_ideal):
    n = len(B_ideal)
    residuals = []
    for i in range(n):
        ax_i = sum(A_ideal[i][j] * x[j] for j in range(n))
        residuals.append(ax_i - B_ideal[i])
    return residuals


precisions = [2,4,6,10]
A_ideal, B_ideal = get_ideal_system()


header = f"{'Знаки':<6} | {'Метод':<9} | {'x1, x2, x3, x4':<55} | {'Невязка':<80}"
print(header)
print("-" * 160)

for p in precisions:
    A_round, B_round = get_matrix_with_precision(p)

    try:
        x_c = solve_gauss_classic(A_round, B_round)
        x_c_str = " ".join([f"{val:<13.7f}" for val in x_c])

        res_c = calculate_residual_vector(A_ideal, x_c, B_ideal)
        res_c_str = " ".join([f"{r:<19.10e}" for r in res_c])

        print(f"{p:<6} | {'Гаусса':<9} | {x_c_str} | {res_c_str}")
    except Exception as e:
        print(f"{p:<6} | {'Гаусса':<9} | Ошибка: {e:<48} | ----")


    try:
        x_p = solve_gauss_pivot(A_round, B_round)
        x_p_str = " ".join([f"{val:<13.7f}" for val in x_p])

        res_p = calculate_residual_vector(A_ideal, x_p, B_ideal)
        res_p_str = " ".join([f"{r:<19.10e}" for r in res_p])

        print(f"{p:<6} | {'С выбором':<9} | {x_p_str} | {res_p_str}")
    except Exception as e:
        print(f"{p:<6} | {'С выбором':<9} | Ошибка: {e:<48} | ----")

