import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

LAYER_SIZES = [400, 25, 10]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_one(x_input, thetas, layer_sizes):
    a = x_input.reshape(-1, 1)
    num_layers = len(layer_sizes)
    
    for l in range(num_layers - 1):
        a = np.vstack([np.ones((1, 1)), a])
        z = np.dot(thetas[l], a)
        a = sigmoid(z)
    
    return np.argmax(a)

def predict_batch(X, thetas, layer_sizes):
    predictions = []
    for i in range(X.shape[0]):
        predictions.append(predict_one(X[i], thetas, layer_sizes))
    return np.array(predictions)


def load_theta_from_txt(filepath):
    """Загрузка матрицы Theta из текстового файла"""
    return np.loadtxt(filepath)


# 1. ЗАГРУЗКА ДАННЫХ
print("Загрузка данных...")
data = scipy.io.loadmat("/home/zerd/all/YPM-IskIm/iskin/labe9/data.mat")
X = data["X"]
y = data["y"]

if y.ndim == 1 or y.shape[1] == 1:
    y = y.reshape(-1, 1)

# Преобразуем метки (10 → 0)
y[y == 10] = 0
y_true = y.flatten()

print(f"Загружено {X.shape[0]} изображений\n")

# 2. ЗАГРУЗКА ПРЕДОБУЧЕННЫХ ВЕСОВ ИЗ ТЕКСТОВЫХ ФАЙЛОВ
print("Загрузка предобученных весов из текстовых файлов...")

# Укажите правильные пути к вашим txt файлам
Theta1 = load_theta_from_txt("/home/zerd/all/YPM-IskIm/iskin/labe7/Theta1.txt")
Theta2 = load_theta_from_txt("/home/zerd/all/YPM-IskIm/iskin/labe7/Theta2.txt")

thetas = [Theta1, Theta2]

print(f"Theta1 shape: {Theta1.shape}")
print(f"Theta2 shape: {Theta2.shape}\n")

# 3. ПРЕДСКАЗАНИЕ
print("Предсказание...")
y_pred = predict_batch(X, thetas, LAYER_SIZES)

# 4. ОЦЕНКА
accuracy = np.mean(y_pred == y_true) * 100
print(f"\nТОЧНОСТЬ НА ВСЕХ ДАННЫХ: {accuracy:.2f}%")

print("\nОТЧЕТ ПО КАЖДОМУ КЛАССУ:")
print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))

# 5. ВИЗУАЛИЗАЦИЯ (10 случайных примеров)
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