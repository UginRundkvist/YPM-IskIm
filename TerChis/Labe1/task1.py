import math
from decimal import Decimal, getcontext, ROUND_HALF_UP


A_matrix = [
    [math.sqrt(5), 2*math.sqrt(3), 0.5, -1.4],
    [-math.sqrt(3), -6/math.sqrt(5), 3.2, 2.5],
    [-1.1, 4.2, -3.2, 0.8],
    [7.9, -5.2, 0.3, 2.9]
]
b_vector = [4.5, 0.7, 0.9, 6.2]

class GaussInterrupt(Exception):
    pass

def find_exact_solution():
    A_ex = [row[:] for row in A_matrix]
    b_ex = b_vector[:]
    n = 4
    
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(A_ex[r][i]))
        if max_row != i:
            A_ex[i], A_ex[max_row] = A_ex[max_row], A_ex[i]
            b_ex[i], b_ex[max_row] = b_ex[max_row], b_ex[i]
        
        for j in range(i+1, n):
            factor = A_ex[j][i] / A_ex[i][i]
            for k in range(i, n):
                A_ex[j][k] -= factor * A_ex[i][k]
            b_ex[j] -= factor * b_ex[i]
    
    x = [0]*n
    for i in range(n-1, -1, -1):
        s = sum(A_ex[i][j]*x[j] for j in range(i+1, n))
        x[i] = (b_ex[i] - s) / A_ex[i][i]
    return x

class Solver:
    def __init__(self, prec):
        self.prec = prec 
        if prec > 0:
            getcontext().prec = prec + 10
    
    def _round(self, v):
        if self.prec <= 0:
            return v
        
        if abs(v) < 1e-15:
            return 0.0
        
        if math.isnan(v) or math.isinf(v):
            return 0.0
        
        try:
            # Преобразуем в строку с фиксированной точкой
            d = Decimal(f"{v:.15f}").quantize(
                Decimal(f'1e-{self.prec}'), 
                rounding=ROUND_HALF_UP
            )
            return float(d)
        except:
            return v
    
    def gauss(self, use_pivot=True):
        n = 4
        A = [row[:] for row in A_matrix]
        b = self.b_original[:]
        
        # Прямой ход
        for i in range(n):
            if use_pivot:
                max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
                if max_row != i:
                    A[i], A[max_row] = A[max_row], A[i]
                    b[i], b[max_row] = b[max_row], b[i]
            
            if abs(A[i][i]) < 1e-14:
                raise GaussInterrupt(f"Нулевой элемент A[{i}][{i}]={A[i][i]:.2e}")
            
            for j in range(i+1, n):
                if abs(A[i][i]) < 1e-14:
                    raise GaussInterrupt(f"Деление на ноль на шаге {i}")
                
                factor = A[j][i] / A[i][i]
                factor = self._round(factor)
                
                for k in range(i, n):
                    A[j][k] = A[j][k] - factor * A[i][k]
                    A[j][k] = self._round(A[j][k])
                
                b[j] = b[j] - factor * b[i]
                b[j] = self._round(b[j])
        
        # Обратный ход
        x = [0]*n
        for i in range(n-1, -1, -1):
            if abs(A[i][i]) < 1e-14:
                raise GaussInterrupt(f"Нулевой элемент в обратном ходе A[{i}][{i}]={A[i][i]:.2e}")
            
            s = 0
            for j in range(i+1, n):
                s += A[i][j] * x[j]
            s = self._round(s)
            
            x[i] = (b[i] - s) / A[i][i]
            x[i] = self._round(x[i])
        
        return x
    
    def check(self, x):
        residuals = []
        for i in range(4):
            s = 0
            for j in range(4):
                s += A_matrix[i][j] * x[j]
            residuals.append(abs(s - b_vector[i]))
        return max(residuals)

def main():
    exact = find_exact_solution()
    
    print("СИСТЕМА УРАВНЕНИЙ")
    for i in range(4):
        print(f"{i+1}: {A_matrix[i][0]:8.5f} x1 + {A_matrix[i][1]:8.5f} x2 + "
              f"{A_matrix[i][2]:8.5f} x3 + {A_matrix[i][3]:8.5f} x4 = {b_vector[i]:5.2f}")
    
    print(f"\nТОЧНОЕ РЕШЕНИЕ (машинная точность):")
    for i, val in enumerate(exact):
        print(f"  x{i+1} = {val:.12f}")
    
    print("РЕЗУЛЬТАТЫ РЕШЕНИЯ")
    
    for prec in [2, 4, 6, 10, -1]:
        prec_name = f"{prec} знаков" if prec > 0 else "машинная точность"
        print(f"\nТочность: {prec_name}")
        
        solver = Solver(prec)
        solver.b_original = b_vector[:]
        
        try:
            x1 = solver.gauss(use_pivot=False)
            err1 = solver.check(x1)
            
            print("Метод Гаусса (без выбора):")
            for i in range(4):
                if prec > 0:
                    print(f"  x{i+1} = {x1[i]:.{prec}f}")
                else:
                    print(f"  x{i+1} = {x1[i]:.10f}")
            print(f"  Макс. невязка = {err1:.2e}")
        except GaussInterrupt as e:
            print(f"Метод Гаусса (без выбора): ПРЕРЫВАНИЕ - {e}")
        
        try:
            x2 = solver.gauss(use_pivot=True)
            err2 = solver.check(x2)
            
            print("\nМетод Гаусса (с выбором главного элемента):")
            for i in range(4):
                if prec > 0:
                    print(f"  x{i+1} = {x2[i]:.{prec}f}")
                else:
                    print(f"  x{i+1} = {x2[i]:.10f}")
            print(f"  Макс. невязка = {err2:.2e}")
        except GaussInterrupt as e:
            print(f"Метод Гаусса (с выбором): ПРЕРЫВАНИЕ - {e}")

if __name__ == "__main__":
    main()