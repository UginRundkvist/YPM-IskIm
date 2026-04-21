import math
from decimal import Decimal, getcontext, ROUND_HALF_UP

# Матрица A
A = [
    [math.sqrt(5), 2 * math.sqrt(3), 0.5, -1.4],
    [-math.sqrt(3), -6 / math.sqrt(5), 3.2, 2.5],
    [-1.1, 4.2, -3.2, 0.8],
    [7.9, -5.2, 0.3, 2.9]
]

# Вектор правой части b
b = [4.5, 0.7, 0.9, 6.2]


def find_exact_solution():
    A_exact = [row[:] for row in A]
    b_exact = b[:]
    n = 4
    
    for i in range(n):
        max_row = i
        max_val = abs(A_exact[i][i])
        for k in range(i+1, n):
            if abs(A_exact[k][i]) > max_val:
                max_val = abs(A_exact[k][i])
                max_row = k
        if max_row != i:
            A_exact[i], A_exact[max_row] = A_exact[max_row], A_exact[i]
            b_exact[i], b_exact[max_row] = b_exact[max_row], b_exact[i]
        
        for j in range(i+1, n):
            factor = A_exact[j][i] / A_exact[i][i]
            for k in range(i, n):
                A_exact[j][k] -= factor * A_exact[i][k]
            b_exact[j] -= factor * b_exact[i]
    
    # Обратный ход
    x = [0] * n
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += A_exact[i][j] * x[j]
        x[i] = (b_exact[i] - s) / A_exact[i][i]
    return x


def print_system():
    print("ИСХОДНАЯ СИСТЕМА УРАВНЕНИЙ")
    for i in range(4):
        print(f"{A[i][0]:8.4f} x1 + {A[i][1]:8.4f} x2 + {A[i][2]:8.4f} x3 + {A[i][3]:8.4f} x4 = {b[i]:8.4f}")

class PrecisionSolver:
    def __init__(self, A, b, decimal_places):
        self.A_original = [row[:] for row in A]
        self.b_original = b[:]
        self.decimal_places = decimal_places
        if decimal_places > 0:
            getcontext().prec = decimal_places + 5
        
    def _round(self, value):
        if self.decimal_places == -1:
            return value
        try:
            # Преобразуем в строку с фиксированной точностью
            d = Decimal(str(value)).quantize(Decimal(f'1e-{self.decimal_places}'), rounding=ROUND_HALF_UP)
            return float(d)
        except:
            return value
    
    def gauss_standard(self):
        n = 4
        A = [row[:] for row in self.A_original]
        b = self.b_original[:]
        
        # Прямой ход
        for i in range(n):
            if abs(A[i][i]) < 1e-12:
                raise Exception(f"Нулевой диагональный элемент A[{i}][{i}] = {A[i][i]}")
            
            for j in range(i+1, n):
                factor = A[j][i] / A[i][i]
                for k in range(i, n):
                    A[j][k] = A[j][k] - factor * A[i][k]
                b[j] = b[j] - factor * b[i]
        
        # Обратный ход
        x = [0] * n
        for i in range(n-1, -1, -1):
            s = 0
            for j in range(i+1, n):
                s += A[i][j] * x[j]
            x[i] = (b[i] - s) / A[i][i]
        
        # Округляем результат после всех вычислений
        if self.decimal_places > 0:
            x = [self._round(v) for v in x]
        
        return x
    
    def gauss_pivot(self):
        """Метод Гаусса с выбором главного элемента"""
        n = 4
        A = [row[:] for row in self.A_original]
        b = self.b_original[:]
        
        # Прямой ход с выбором главного элемента
        for i in range(n):
            # Поиск максимального элемента в столбце
            max_row = i
            max_val = abs(A[i][i])
            for k in range(i+1, n):
                if abs(A[k][i]) > max_val:
                    max_val = abs(A[k][i])
                    max_row = k
            
            if max_val < 1e-12:
                raise Exception(f"Все элементы в столбце {i} близки к нулю")
            
            # Меняем строки местами
            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i]
                b[i], b[max_row] = b[max_row], b[i]
            
            for j in range(i+1, n):
                factor = A[j][i] / A[i][i]
                for k in range(i, n):
                    A[j][k] = A[j][k] - factor * A[i][k]
                b[j] = b[j] - factor * b[i]
        
        # Обратный ход
        x = [0] * n
        for i in range(n-1, -1, -1):
            s = 0
            for j in range(i+1, n):
                s += A[i][j] * x[j]
            x[i] = (b[i] - s) / A[i][i]
        
        # Округляем результат после всех вычислений
        if self.decimal_places > 0:
            x = [self._round(v) for v in x]
        
        return x
    
    def check_solution(self, x):
        n = 4
        residuals = []
        for i in range(n):
            s = 0
            for j in range(n):
                s += self.A_original[i][j] * x[j]
            residuals.append(s - self.b_original[i])
        return residuals


def main():
    print("РЕШЕНИЕ СЛАУ МЕТОДАМИ ГАУССА")
    print("Вариант 16")
    
    print_system()
    
    # Находим точное решение для сравнения
    exact = find_exact_solution()
    print("\n" + "="*60)
    print("ЭТАЛОННОЕ РЕШЕНИЕ (с машинной точностью)")
    print("="*60)
    for i in range(4):
        print(f"x{i+1} = {exact[i]:.10f}")
    
    # Точности для исследования
    precisions = [2, 4, 6, 10, -1]  # -1 = машинная точность
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ РЕШЕНИЯ")
    print("="*60)
    
    # Для каждой точности
    for prec in precisions:
        print(f"\n--- Точность: {prec} знаков после запятой" if prec > 0 else "\n--- Точность: машинная точность")
        print("-" * 50)
        
        # Метод Гаусса без выбора
        try:
            solver1 = PrecisionSolver(A, b, prec)
            x1 = solver1.gauss_standard()
            res1 = solver1.check_solution(x1)
            max_res1 = max(abs(r) for r in res1)
            
            print("Метод Гаусса (без выбора):")
            if prec > 0:
                print(f"  x1 = {x1[0]:.{prec}f}")
                print(f"  x2 = {x1[1]:.{prec}f}")
                print(f"  x3 = {x1[2]:.{prec}f}")
                print(f"  x4 = {x1[3]:.{prec}f}")
            else:
                print(f"  x1 = {x1[0]:.10f}")
                print(f"  x2 = {x1[1]:.10f}")
                print(f"  x3 = {x1[2]:.10f}")
                print(f"  x4 = {x1[3]:.10f}")
            print(f"  Макс. невязка = {max_res1:.2e}")
        except Exception as e:
            print(f"Метод Гаусса (без выбора): ОШИБКА - {e}")
        
        # Метод Гаусса с выбором
        try:
            solver2 = PrecisionSolver(A, b, prec)
            x2 = solver2.gauss_pivot()
            res2 = solver2.check_solution(x2)
            max_res2 = max(abs(r) for r in res2)
            
            print("\nМетод Гаусса (с выбором главного элемента):")
            if prec > 0:
                print(f"  x1 = {x2[0]:.{prec}f}")
                print(f"  x2 = {x2[1]:.{prec}f}")
                print(f"  x3 = {x2[2]:.{prec}f}")
                print(f"  x4 = {x2[3]:.{prec}f}")
            else:
                print(f"  x1 = {x2[0]:.10f}")
                print(f"  x2 = {x2[1]:.10f}")
                print(f"  x3 = {x2[2]:.10f}")
                print(f"  x4 = {x2[3]:.10f}")
            print(f"  Макс. невязка = {max_res2:.2e}")
        except Exception as e:
            print(f"Метод Гаусса (с выбором): ОШИБКА - {e}")


if __name__ == "__main__":
    main()