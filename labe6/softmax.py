#C:/Users/1/Desktop/IskIn/YPM-IskIm/labe6/data.mat
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Загрузка данных
data = sio.loadmat('C:/Users/1/Desktop/IskIn/YPM-IskIm/labe6/data.mat')
X = data['X'] 
y = data['y'] 
y[y == 10] = 0

# Добавляем столбец единиц (bias)
X = np.hstack([np.ones((X.shape[0], 1)), X])  

# Функция softmax
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)  # Численная стабильность
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# One-hot encoding для меток
def one_hot(y, num_labels):
    m = y.shape[0]
    y_onehot = np.zeros((m, num_labels))
    for i in range(m):
        y_onehot[i, int(y[i])] = 1
    return y_onehot

# Функция стоимости (категориальная кросс-энтропия + регуляризация)
def cost_softmax(theta, X, y_onehot, lam, num_labels):
    m = X.shape[0]
    theta = theta.reshape(num_labels, X.shape[1])
    
    h = softmax(X @ theta.T)
    h = np.clip(h, 1e-15, 1 - 1e-15)  # Защита от log(0)
    
    # Основная часть ошибки
    J = (-1/m) * np.sum(y_onehot * np.log(h))
    
    # Регуляризация (исключая bias - первый столбец)
    reg = (lam/(2*m)) * np.sum(theta[:, 1:]**2)
    
    return J + reg

# Градиент для softmax
def grad_softmax(theta, X, y_onehot, lam, num_labels):
    m = X.shape[0]
    theta = theta.reshape(num_labels, X.shape[1])
    
    h = softmax(X @ theta.T)
    h = np.clip(h, 1e-15, 1 - 1e-15)
    
    # Градиент без регуляризации
    grad = (-1/m) * ((y_onehot - h).T @ X)
    
    # Добавляем регуляризацию (исключая bias)
    grad[:, 1:] += (lam/m) * theta[:, 1:]
    
    return grad.flatten()

# Функция обучения softmax (вместо one_vs_all)
def train_softmax(X, y, num_labels, lam):
    y_onehot = one_hot(y, num_labels)
    n_features = X.shape[1]
    
    # Инициализация (все нули)
    initial_theta = np.zeros(num_labels * n_features)
    
    # Оптимизация
    result = minimize(fun=cost_softmax, 
                      x0=initial_theta,
                      args=(X, y_onehot, lam, num_labels),
                      method='TNC',
                      jac=grad_softmax,
                      options={'maxiter': 400})
    
    # Возвращаем theta в виде матрицы (num_labels x n_features)
    all_theta = result.x.reshape(num_labels, X.shape[1])
    print("Обучение Softmax завершено")
    return all_theta

# Функция предсказания
def predict_softmax(all_theta, X):
    return np.argmax(softmax(X @ all_theta.T), axis=1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y.flatten(), test_size=0.2, random_state=42
)

print("Обучение Softmax классификатора...")
# Обучаем (10 классов)
all_theta = train_softmax(X_train, y_train, 10, lam=0.001)

# Предсказание на тестовой выборке
y_pred = predict_softmax(all_theta, X_test)
accuracy = np.mean(y_pred == y_test) * 100
print(f"\nТОЧНОСТЬ: {accuracy:.2f}%")

print("ОТЧЕТ ПО КАЖДОМУ КЛАССУ:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# Визуализация результатов на тестовой выборке
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(X_test))
    ax.imshow(X_test[idx, 1:].reshape(20, 20).T, cmap='gray')
    ax.set_title(f'True: {int(y_test[idx])}\nPred: {y_pred[idx]}', 
                 color='green' if y_test[idx]==y_pred[idx] else 'red')
    ax.axis('off')
plt.tight_layout()
plt.show()