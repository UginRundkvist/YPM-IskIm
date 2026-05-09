#C:/Users/1/Desktop/IskIn/YPM-IskIm/labe6/data.mat
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data = sio.loadmat('/home/zerd/all/YPM-IskIm/labe6/data.mat')
X = data['X'] 
y = data['y'] 
y[y == 10] = 0

X = np.hstack([np.ones((X.shape[0], 1)), X])  

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, lam):
    m = len(y)
    h = sigmoid(X @ theta)
    h = np.clip(h, 1e-15, 1 - 1e-15)
    reg = (lam/(2*m)) * np.sum(theta[1:]**2)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h)) + reg

def grad(theta, X, y, lam):
    m = len(y)
    h = sigmoid(X @ theta)
    h = np.clip(h, 1e-15, 1 - 1e-15)
    g = (1/m) * (X.T @ (h - y))
    g[1:] += (lam/m) * theta[1:]
    return g

def one_vs_all(X, y, num_labels, lam):
    all_theta = np.zeros((num_labels, X.shape[1]))
    for i in range(num_labels):
        print(f"Класс {i} обучен")
        result = minimize(fun=cost, x0=np.zeros(X.shape[1]), 
                         args=(X, (y==i).astype(int), lam),
                         method='TNC', jac=grad,  
                         options={'maxiter': 400})  
        all_theta[i] = result.x
    return all_theta

def predict(all_theta, X):
    return np.argmax(sigmoid(X @ all_theta.T), axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y.flatten(), test_size=0.2, random_state=42
)

print("Обучение 10 классификаторов (один против всех)...")
all_theta = one_vs_all(X_train, y_train, 10, lam=0.001)

# Предсказание на тестовой выборке
y_pred = predict(all_theta, X_test)
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