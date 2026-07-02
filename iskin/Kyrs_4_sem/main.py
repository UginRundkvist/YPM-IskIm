import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import os

np.random.seed(42)
random.seed(42)

# TRAINING_MODE = "mini_batch"
LEARNING_RATE = 0.01
BATCH_SIZE = 128
EPOCHS = 30

ARCHITECTURES = [[32],[64], [64, 32],[128, 64], [128, 64, 32]]

ACTIVATIONS = [ "relu","tanh", "sigmoid"]

INITIALIZATION = { "relu": "he", "tanh": "xavier", "sigmoid":"he", "sigmoid":"xavier"}

DROPOUT_RATES = [ 0.0, 0.1, 0.2, 0.3]

df = pd.read_csv('C:/Users/1/Desktop/IskIn/YPM-IskIm/iskin/Kyrs_4_sem/WineQT.csv', sep=';')

print(f"\nИсходный размер датасета: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")

if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

print("\nРаспределение оценок качества:")
quality_counts = df['quality'].value_counts().sort_index()
for q, count in quality_counts.items():
    print(f"  Оценка {q}: {count} образцов")

class_counts = df['quality'].value_counts()
rare_classes = class_counts[class_counts < 31].index.tolist()

if rare_classes:
    print(f"Найдены редкие классы: {rare_classes}")
    df = df[~df['quality'].isin(rare_classes)]

X_raw = df.drop('quality', axis=1).values.astype(float)
y_raw = df['quality'].values.astype(int)

unique_classes = np.unique(y_raw)
n_classes = len(unique_classes)

def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_shifted = y_raw - min(unique_classes)
y_onehot = to_one_hot(y_shifted, n_classes)

X_train_raw, X_temp, y_train_labels, y_temp = train_test_split(
        X_raw, y_shifted, test_size=0.30, random_state=42
    )
X_val_raw, X_test_raw, y_val_labels, y_test_labels = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

y_train = to_one_hot(y_train_labels, n_classes)
y_val = to_one_hot(y_val_labels, n_classes)
y_test = to_one_hot(y_test_labels, n_classes)

print(f"Train: {X_train_raw.shape[0]} образцов")
print(f"Validation: {X_val_raw.shape[0]} образцов")
print(f"Test: {X_test_raw.shape[0]} образцов")

mean_train = np.mean(X_train_raw, axis=0)
std_train = np.std(X_train_raw, axis=0)
std_train[std_train == 0] = 1

def scale(X, mean_val, std_val):
    return (X - mean_val) / std_val

X_train = scale(X_train_raw, mean_train, std_train)
X_val = scale(X_val_raw, mean_train, std_train)
X_test = scale(X_test_raw, mean_train, std_train)


def softmax(z):
    shift_z = z - np.max(z, axis=1, keepdims=True)
    return np.exp(shift_z) / np.sum(np.exp(shift_z), axis=1, keepdims=True)

def get_activation(name):
    if name == 'relu':
        return lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)
    elif name == 'tanh':
        return lambda z: np.tanh(z), lambda z: 1.0 - np.square(np.tanh(z))
    elif name == 'sigmoid':
        act = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return act, lambda z: act(z) * (1.0 - act(z))

def init_weights(method, d_in, d_out):
    if method == 'he':
        return np.random.randn(d_in, d_out) * np.sqrt(2.0 / d_in) 
    elif method == 'xavier':
        return np.random.randn(d_in, d_out) * np.sqrt(1.0 / d_in)

class CombinatoricNeuralNetwork:
    def __init__(self, dims, init_method, activation_name, dropout_keep_prob):
        self.keep_prob = dropout_keep_prob
        self.weights = []
        self.biases = []
        self.act, self.act_deriv = get_activation(activation_name)
        
        for i in range(len(dims) - 1):
            self.weights.append(init_weights(init_method, dims[i], dims[i + 1]))
            self.biases.append(np.zeros((1, dims[i + 1])))

    def forward(self, X, training=True):
        self.activation = [X]
        self.pre_activation = []
        self.D = []
        
        for i in range(len(self.weights) - 1):
            pre_activation = np.dot(self.activation[-1], self.weights[i]) + self.biases[i]
            activation = self.act(pre_activation)
            
            if training and self.keep_prob < 1.0:
                d = np.random.rand(*activation.shape) < self.keep_prob
                activation = (activation * d) / self.keep_prob
                self.D.append(d)
            else:
                self.D.append(None)
            
            self.pre_activation.append(pre_activation)
            self.activation.append(activation)
        
        self.pre_activation.append(np.dot(self.activation[-1], self.weights[-1]) + self.biases[-1])
        self.activation.append(softmax(self.pre_activation[-1]))
        
        return self.activation[-1]

    def compute_loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(np.clip(y_pred, 1e-15, 1.0 - 1e-15))) / y_true.shape[0]
    
    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]
        self.dWs = [None] * len(self.weights)
        self.dbs = [None] * len(self.weights)
        
        grad_pre_activation = y_pred - y_true
        
        for i in range(len(self.weights) - 1, -1, -1):
            self.dWs[i] = np.dot(self.activation[i].T, grad_pre_activation) / m
            self.dbs[i] = np.sum(grad_pre_activation, axis=0, keepdims=True) / m
            
            if i > 0:
                grad_activation = np.dot(grad_pre_activation, self.weights[i].T)
                if self.keep_prob < 1.0 and self.D[i - 1] is not None:
                    grad_activation = (grad_activation * self.D[i - 1]) / self.keep_prob
                    
                grad_pre_activation = grad_activation * self.act_deriv(self.pre_activation[i - 1])

    def update_parameters(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.dWs[i]
            self.biases[i] -= lr * self.dbs[i]

def accuracy(y_true, y_pred):
    return np.mean(
        np.argmax(y_true, axis=1) ==  np.argmax(y_pred, axis=1) )

def run_experiment(  X_train, y_train, X_val, y_val,architecture,activation, dropout, track_losses=False):

    init_method = INITIALIZATION[activation]
    dims = [X_train.shape[1]] + architecture + [y_train.shape[1]]

    nn = CombinatoricNeuralNetwork(
        dims=dims,
        init_method=init_method,
        activation_name=activation,
        dropout_keep_prob=1.0 - dropout
    )

    train_losses = []
    val_losses = []

    num_samples = X_train.shape[0]

    train_acc = []
    val_acc = []
    
    for _ in range(EPOCHS):
        indices = np.random.permutation(num_samples)

        X_shuffle = X_train[indices]
        y_shuffle = y_train[indices]

        for i in range(0, num_samples, BATCH_SIZE):
            X_batch = X_shuffle[i:i + BATCH_SIZE]
            y_batch = y_shuffle[i:i + BATCH_SIZE]
            
            prediction = nn.forward(X_batch, training=True)

            nn.backward(X_batch, y_batch, prediction)

            nn.update_parameters(LEARNING_RATE)

        if track_losses:
            train_prediction = nn.forward(X_train, training=False)
            val_prediction = nn.forward(X_val, training=False)

            train_losses.append(
                nn.compute_loss(y_train, train_prediction)
            )

            val_losses.append(
                nn.compute_loss(y_val, val_prediction)
            )

            train_acc.append(
                accuracy(y_train, train_prediction)
            )

            val_acc.append(
                accuracy(y_val, val_prediction)
            )
    train_prediction = nn.forward(X_train, training=False)
    val_prediction = nn.forward(X_val, training=False)

    train_losses.append(nn.compute_loss(y_train, train_prediction))
    val_losses.append(nn.compute_loss(y_val, val_prediction))

    train_acc.append(accuracy(y_train, train_prediction))
    val_acc.append(accuracy(y_val, val_prediction))
            
    return nn, train_losses, val_losses, train_acc, val_acc


def architecture_experiment():
    results = []
    for architecture in ARCHITECTURES:

        print(f"\nАрхитектура: {architecture}")
        model, _, _, _, _ = run_experiment(  X_train,  y_train,  X_val,  y_val,
            architecture=architecture,
            activation="relu",
            dropout=0.2,
            track_losses=False
        )

        y_pred = np.argmax(model.forward(X_test, training=False), axis=1)

        accuracy = np.mean(  y_pred == np.argmax(y_test, axis=1)) * 100

        results.append({
            "architecture": architecture,
            "accuracy": accuracy
        })

        print(f"Точность: {accuracy:.2f}%")

    best = max(results, key=lambda x: x["accuracy"])

    print(f"Архитектура: {best['architecture']}")
    print(f"Accuracy: {best['accuracy']:.2f}%")

    return best["architecture"]

def activation_experiment(best_architecture):
    results = []
    print(f"{'Активация':<15}{'Accuracy':>15}")

    for activation in ACTIVATIONS:

        model, _, _, _, _ = run_experiment(
            X_train,
            y_train,
            X_val,
            y_val,
            architecture=best_architecture,
            activation=activation,
            dropout=0.2,
            track_losses=False
        )

        y_pred = np.argmax( model.forward(X_test, training=False),  axis=1 )

        accuracy = np.mean(  y_pred == np.argmax(y_test, axis=1) ) * 100

        results.append((activation, accuracy))

        print(f"{activation:<15}{accuracy:>13.2f}%")

    best_activation, best_accuracy = max(
        results,
        key=lambda x: x[1]
    )

    print(f"\nЛучшая функция активации: {best_activation}")
    print(f"Точность: {best_accuracy:.2f}%")

    return best_activation

def dropout_experiment(best_architecture, best_activation):
    results = []
    print(f"{'Dropout':<15}{'Accuracy':>15}")

    for dropout in DROPOUT_RATES:
        model, _, _, _, _ = run_experiment(
            X_train,
            y_train,
            X_val,
            y_val,
            architecture=best_architecture,
            activation=best_activation,
            dropout=dropout,
            track_losses=False
        )

        y_pred = np.argmax(model.forward(X_test, training=False),axis=1 )

        accuracy = np.mean( y_pred == np.argmax(y_test, axis=1) ) * 100

        results.append((dropout, accuracy))

        print(f"{dropout:<15}{accuracy:>13.2f}%")

    best_dropout, best_accuracy = max(
        results,
        key=lambda x: x[1]
    )

    print(f"\nЛучшее значение Dropout: {best_dropout}")
    print(f"Точность: {best_accuracy:.2f}%")

    return best_dropout



best_architecture = architecture_experiment()
best_activation = activation_experiment(best_architecture)
best_dropout = dropout_experiment(best_architecture, best_activation)

best_model, train_loss, val_loss, train_acc, val_acc = run_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    architecture=best_architecture,
    activation=best_activation,
    dropout=best_dropout,
    track_losses=True
)


#графики

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("График изменения функции потерь")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def plot_accuracy(train_acc, val_acc):
    plt.figure(figsize=(8,5))

    plt.plot(train_acc, label="Train Accuracy", linewidth=2)
    plt.plot(val_acc, label="Validation Accuracy", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Точность модели")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_losses(train_loss, val_loss)
plot_accuracy(train_acc, val_acc)