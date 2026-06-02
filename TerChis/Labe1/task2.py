A = [ [1, 7, 1, 5],
    [5, 1, -1, 1],
    [-1, 2, -2, 19],
    [1, 2, 9, -11] ]

b = [7, 17, 18, -17]

def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= off:
            return False
    return True

def make_diagonally_dominant(A, b):
    n = len(A)
    A_new = []
    b_new = []
    used = [False] * n
    for i in range(n):
        best_row = -1
        best_ratio = -1
        for row in range(n):

            if used[row]:
                continue

            diag = abs(A[row][i])
            off = sum(abs(A[row][j]) for j in range(n) if j != i)

            if diag > off:
                best_row = row
                break

            ratio = diag / (off + 1e-12)

            if ratio > best_ratio:
                best_ratio = ratio
                best_row = row

        used[best_row] = True

        A_new.append(A[best_row])
        b_new.append(b[best_row])

    return A_new, b_new


def seidel_sor(A, b, omega, eps=1e-3, max_iter=500):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    d = [0] * n

    for i in range(n):
        if abs(A[i][i]) < 1e-12:
            print("Нулевой диагональный элемент.")
            return None, 0
        d[i] = b[i] / A[i][i]
        for j in range(n):
            if j != i:
                C[i][j] = -A[i][j] / A[i][i]

    x = [0] * n

    for iteration in range(1, max_iter + 1):
        x_prev = x[:]
        max_err = 0
        for i in range(n):
            s1 = sum(C[i][j] * x[j] for j in range(i))
            s2 = sum(C[i][j] * x_prev[j] for j in range(i + 1, n))
            x_new = d[i] + s1 + s2
            x[i] = omega * x_new + (1 - omega) * x_prev[i]
            max_err = max(max_err, abs(x[i] - x_prev[i]))

        if max_err < eps:
            return x, iteration

    return x, max_iter

def residual(A, b, x):

    n = len(A)

    return max(
        abs(sum(A[i][j] * x[j] for j in range(n)) - b[i])
        for i in range(n)
    )

print("ИСХОДНАЯ СИСТЕМА:\n")

for i in range(4):
    print(
        f"{A[i][0]:3d}x1 + "
        f"{A[i][1]:3d}x2 + "
        f"{A[i][2]:3d}x3 + "
        f"{A[i][3]:3d}x4 = "
        f"{b[i]:3d}" )


A_new, b_new = make_diagonally_dominant(A, b)

print("\nПРЕОБРАЗОВАННАЯ СИСТЕМА:\n")

for i in range(4):
    print(
        f"{A_new[i][0]:3d}x1 + "
        f"{A_new[i][1]:3d}x2 + "
        f"{A_new[i][2]:3d}x3 + "
        f"{A_new[i][3]:3d}x4 = "
        f"{b_new[i]:3d}")

if is_diagonally_dominant(A_new):
    print("\nМатрица обладает строгим диагональным преобладанием.\n")
else:
    print("\nСтрогое диагональное преобладание получить")
    print("перестановкой строк НЕ удалось.\n")
    print("Используется наиболее подходящая перестановка.\n")


print("ИССЛЕДОВАНИЕ РЕЛАКСАЦИИ (0 < ω < 1)\n")

print("  ω   | Итерации |"
      "     x1    |"
      "    x2     |"
      "     x3    |"
      "     x4     |"
      " Невязка")

results = []

omegas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for omega in omegas:
    x, iterations = seidel_sor(A_new, b_new, omega, eps=1e-3)

    if x is None:
        print(f"{omega:.1f} | не сходится")
        continue

    res = residual(A, b, x)

    results.append((omega, iterations, x, res))

    print(
        f" {omega:.1f}  |"
        f" {iterations:3d}|"
        f" {x[0]:11.6f} |"
        f" {x[1]:11.6f} |"
        f" {x[2]:11.6f} |"
        f" {x[3]:11.6f} |"
        f" {res:.2e}"
    )

if results:

    best = min(results, key=lambda r: r[1])

    print("\nЛУЧШИЙ РЕЗУЛЬТАТ:\n")

    print(f"Оптимальный параметр ω = {best[0]}")
    print(f"Количество итераций = {best[1]}\n")

    print("РЕШЕНИЕ:\n")

    for i in range(4):
        print(f"x{i+1} = {best[2][i]:.6f}")
    print(f"\nНевязка = {best[3]:.2e}")