import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

INPUT_SIZE = 400
HIDDEN_SIZE = 25
OUTPUT_SIZE = 10

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_one(x_input, theta1, theta2):
    a1 = x_input.reshape(-1, 1)
    
    a1 = np.vstack([np.ones((1, 1)), a1]) 
    
    z2 = np.dot(theta1, a1) 
    a2 = sigmoid(z2)
    
    a2 = np.vstack([np.ones((1, 1)), a2]) 
    
    z3 = np.dot(theta2, a2) 
    a3 = sigmoid(z3) 
    
    return (np.argmax(a3) + 1) % 10

def predict_batch(X, theta1, theta2):
    predictions = []
    for i in range(X.shape[0]):
        predictions.append(predict_one(X[i], theta1, theta2))
    return np.array(predictions)

print("ЗАГРУЗКА ДАННЫХ И ВЕСОВ")
print(f"Архитектура сети: {INPUT_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}")
print()

try:
    data = scipy.io.loadmat("/home/zerd/all/YPM-IskIm/iskin/labe7/data.mat")
except:
    print("Ошибка: файл data.mat не найден!")
    exit(1)

X = data["X"]
y = data["y"]

if y.ndim == 1 or y.shape[1] == 1:
    y = y.reshape(-1, 1)

y[y == 10] = 0

print(f"Всего изображений: {X.shape[0]}")

try:
    theta1 = np.loadtxt("/home/zerd/all/YPM-IskIm/iskin/labe7/Theta1.txt")
    theta2 = np.loadtxt("/home/zerd/all/YPM-IskIm/iskin/labe7/Theta2.txt")
    print("Веса успешно загружены\n")
except:
    print("Ошибка: файлы theta1.txt или theta2.txt не найдены!")
    exit(1)

print("ТЕСТИРОВАНИЕ НЕЙРОННОЙ СЕТИ")

y_pred = predict_batch(X, theta1, theta2)
y_true = y.flatten()

accuracy = np.mean(y_pred == y_true) * 100
print(f"\nТОЧНОСТЬ: {accuracy:.2f}%")

print("\nОТЧЕТ ПО КАЖДОМУ КЛАССУ:")
print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle(f"Предсказания нейронной сети (точность: {accuracy:.1f}%)", fontsize=14)

for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(X))
    ax.imshow(X[idx].reshape(20, 20).T, cmap='gray')
    color = 'green' if y_true[idx] == y_pred[idx] else 'red'
    ax.set_title(f'True: {int(y_true[idx])}\nPred: {y_pred[idx]}', 
                 color=color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()