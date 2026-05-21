import numpy as np

#Загружает параметры theta из файла с заголовком.
def load_theta(path="Theta.txt"):
    theta = np.loadtxt(path, skiprows=1)
    return theta.flatten()

#Предсказания
def predict(x, theta):
    return theta[0] + theta[1] * x

#Основная логика
def main():
    theta = load_theta()
    print(f"Загружены веса: θ0 = {theta[0]:.6f}, θ1 = {theta[1]:.6f}")
    print("Введите значения колличество автомобилей через пробел или запятую.")
    
    while True:
        s = input("> ").strip()
        if not s:
            continue
            
        xs = [float(x) for x in s.replace(',', ' ').split()]
        
        for x in xs:
            y_pred = predict(x, theta)
            print(f"x = {x} → y = {y_pred:.6f}")

if __name__ == "__main__":
    main()