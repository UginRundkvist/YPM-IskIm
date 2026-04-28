# Исходная система
A = [[1,7,1,5],[5,1,-1,1],[-1,2,-2,19],[1,2,9,-11]]
b = [7,17,18,-17]

def make_diagonally_dominant(A, b):
    n = len(A)
    A_new = []
    b_new = []
    used = [False]*n
    
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
            if diag / (off + 1e-10) > best_ratio:
                best_ratio = diag / (off + 1e-10)
                best_row = row
        if best_row == -1:
            best_row = [r for r in range(n) if not used[r]][0]
        used[best_row] = True
        A_new.append(A[best_row])
        b_new.append(b[best_row])
    
    return A_new, b_new

def seidel_sor(A, b, omega, eps=1e-3, max_iter=500):
    n = len(A)
    C = [[0]*n for _ in range(n)]
    d = [0]*n
    for i in range(n):
        if abs(A[i][i]) < 1e-12:
            return None, 0
        d[i] = b[i] / A[i][i]
        for j in range(n):
            if j != i:
                C[i][j] = -A[i][j] / A[i][i]
    
    x = [0]*n
    for it in range(1, max_iter+1):
        x_prev = x[:]
        max_err = 0
        for i in range(n):
            s1 = sum(C[i][j]*x[j] for j in range(i))
            s2 = sum(C[i][j]*x_prev[j] for j in range(i+1, n))
            x_new = d[i] + s1 + s2
            x[i] = omega * x_new + (1-omega) * x_prev[i]
            max_err = max(max_err, abs(x[i] - x_prev[i]))
        if max_err < eps:
            return x, it
    return x, max_iter

def residual(A, b, x):
    return max(abs(sum(A[i][j]*x[j] for j in range(4)) - b[i]) for i in range(4))


print("Исходная система:")
for i in range(4):
    print(f"  {A[i][0]:2d}x1 + {A[i][1]:2d}x2 + {A[i][2]:2d}x3 + {A[i][3]:2d}x4 = {b[i]:2d}")

# Преобразуем
A_new, b_new = make_diagonally_dominant(A, b)

print("\nПреобразованная система: диагонально преобладающая:")
for i in range(4):
    print(f"  {A_new[i][0]:2d}x1 + {A_new[i][1]:2d}x2 + {A_new[i][2]:2d}x3 + {A_new[i][3]:2d}x4 = {b_new[i]:2d}")


print("ИССЛЕДОВАНИЕ РЕЛАКСАЦИИ (0 < ω < 1)")
print("  ω   | Итерации |    x1     |    x2     |    x3     |    x4     | Невязка")

results = []
for omega in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    x, it = seidel_sor(A_new, b_new, omega, eps=1e-3)
    if x is None:
        print(f"{omega:.1f}   | не сходится")
        continue
    res = residual(A, b, x)
    results.append((omega, it, x, res))
    print(f"{omega:.1f}   | {it:5d}    | {x[0]:8.5f} {x[1]:8.5f} {x[2]:8.5f} {x[3]:8.5f} | {res:.2e}")

if results:
    best = min(results, key=lambda r: r[1])
    print("РЕЗУЛЬТАТЫ")
    print(f"Оптимальный ω = {best[0]}  (итераций = {best[1]})")
    print(f"\nРешение:")
    for i in range(4):
        print(f"  x{i+1} = {best[2][i]:.6f}")
    print(f"\nНевязка = {best[3]:.2e}")
