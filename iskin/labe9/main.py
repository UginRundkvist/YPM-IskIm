#C:/Users/1/Desktop/IskIn/YPM-IskIm/iskin/labe9/data.mat
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pickle
import os

LAYER_SIZES = [400, 25, 15, 10]

ALPHA = 1.5
LAMBDA_REG = 1.0
STEPS = 2000

TRAIN_SPLIT = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    a = sigmoid(z)
    return a * (1 - a)

def init_thetas(layer_sizes):
    thetas = []
    epsilon_init = 0.12
    
    for i in range(len(layer_sizes) - 1):
        s_in = layer_sizes[i]
        s_out = layer_sizes[i + 1]
        theta = np.random.rand(s_out, s_in + 1) * 2 * epsilon_init - epsilon_init
        thetas.append(theta)
    
    return thetas

def compute_cost_and_gradients(x_batch, y_batch, thetas, layer_sizes, lambda_reg):
    m = x_batch.shape[0]
    num_layers = len(layer_sizes)
    
    y_onehot = np.zeros((m, layer_sizes[-1]))
    for i in range(m):
        y_onehot[i, int(y_batch[i, 0]) % 10] = 1
    
    a = x_batch.T
    a_cache = []
    z_cache = []
    
    for l in range(num_layers - 1):
        a = np.vstack([np.ones((1, m)), a])
        a_cache.append(a)
        z = np.dot(thetas[l], a)
        z_cache.append(z)
        a = sigmoid(z)
    
    a_cache.append(a)
    h = a
    y = y_onehot.T
    
    cost = np.sum(-y * np.log(h + 1e-15) - (1 - y) * np.log(1 - h + 1e-15)) / m
    
    reg = 0
    for theta in thetas:
        reg += np.sum(theta[:, 1:] ** 2)
    cost += (lambda_reg / (2 * m)) * reg
    
    deltas = [0] * num_layers
    deltas[-1] = h - y
    
    for l in range(num_layers - 2, 0, -1):
        theta = thetas[l]
        theta_no_bias = theta[:, 1:]
        z = z_cache[l - 1]
        delta = np.dot(theta_no_bias.T, deltas[l + 1]) * sigmoid_gradient(z)
        deltas[l] = delta
    
    gradients = []
    for l in range(num_layers - 1):
        d_matrix = np.dot(deltas[l + 1], a_cache[l].T) / m
        reg_grad = (lambda_reg / m) * thetas[l]
        reg_grad[:, 0] = 0
        gradients.append(d_matrix + reg_grad)
    
    return cost, gradients

def train_network(x_train, y_train, thetas, layer_sizes, alpha, lambda_reg, steps):
    for step in range(steps):
        cost, gradients = compute_cost_and_gradients(x_train, y_train, thetas, layer_sizes, lambda_reg)
        for l in range(len(thetas)):
            thetas[l] -= alpha * gradients[l]
    return thetas

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


print("ПОДГОТОВКА ДАННЫХ")
print(f"Архитектура сети: {LAYER_SIZES}")
print()

data = scipy.io.loadmat("/home/zerd/all/YPM-IskIm/iskin/labe9/data.mat")
X = data["X"]
y = data["y"]

if y.ndim == 1 or y.shape[1] == 1:
    y = y.reshape(-1, 1)

y[y == 10] = 0

np.random.seed(42) 
permutations = np.random.permutation(X.shape[0])
X = X[permutations]
y = y[permutations]

X_test, X_train = X[:TRAIN_SPLIT, :], X[TRAIN_SPLIT:, :]
y_test, y_train = y[:TRAIN_SPLIT, :], y[TRAIN_SPLIT:, :]

print(f"Обучающая выборка: {X_train.shape[0]} изображений")
print(f"Тестовая выборка: {X_test.shape[0]} изображений\n")

WEIGHTS_FILE = "weights.pkl"

if os.path.exists(WEIGHTS_FILE):
    print("ЗАГРУЗКА ВЕСОВ")
    with open(WEIGHTS_FILE, 'rb') as f:
        thetas = pickle.load(f)
else:
    print("ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ")
    thetas = init_thetas(LAYER_SIZES)
    thetas = train_network(X_train, y_train, thetas, LAYER_SIZES, ALPHA, LAMBDA_REG, STEPS)
    with open(WEIGHTS_FILE, 'wb') as f:
        pickle.dump(thetas, f)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО, ВЕСА СОХРАНЕНЫ\n")

print("ТЕСТИРОВАНИЕ НЕЙРОННОЙ СЕТИ")

y_pred = predict_batch(X_test, thetas, LAYER_SIZES)
y_true = y_test.flatten()

accuracy = np.mean(y_pred == y_true) * 100
print(f"\nТОЧНОСТЬ: {accuracy:.2f}%")

print("\nОТЧЕТ ПО КАЖДОМУ КЛАССУ:")
print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle(f"Предсказания нейронной сети (точность: {accuracy:.1f}%)", fontsize=14)

for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(X_test))
    ax.imshow(X_test[idx].reshape(20, 20).T, cmap='gray')
    color = 'green' if y_true[idx] == y_pred[idx] else 'red'
    ax.set_title(f'True: {int(y_true[idx])}\nPred: {y_pred[idx]}', 
                 color=color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()