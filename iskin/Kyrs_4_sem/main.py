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
    elif name == 'softmax':
        return softmax, lambda z: softmax(z) * (1.0 - softmax(z))

def init_weights(method, d_in, d_out):
    if method == 'he':
        return np.random.randn(d_in, d_out) * np.sqrt(2.0 / d_in) 
    elif method == 'xavier':
        return np.random.randn(d_in, d_out) * np.sqrt(1.0 / d_in)

class CombinatoricNeuralNetwork:
    def __init__(self, dims, init_method, activation_name, dropout_keep_prob=0.85):
        self.keep_prob = dropout_keep_prob
        self.weights = []
        self.biases = []
        self.act, self.act_deriv = get_activation(activation_name)
        
        for i in range(len(dims) - 1):
            self.weights.append(init_weights(init_method, dims[i], dims[i + 1]))
            self.biases.append(np.zeros((1, dims[i + 1])))

    def forward(self, X, training=True):
        self.A = [X]
        self.Z = []
        self.D = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            a = self.act(z)
            
            if training and self.keep_prob < 1.0:
                d = np.random.rand(*a.shape) < self.keep_prob
                a = (a * d) / self.keep_prob
                self.D.append(d)
            else:
                self.D.append(None)
            
            self.Z.append(z)
            self.A.append(a)
        
        self.Z.append(np.dot(self.A[-1], self.weights[-1]) + self.biases[-1])
        self.A.append(softmax(self.Z[-1]))
        
        return self.A[-1]

    def compute_loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(np.clip(y_pred, 1e-15, 1.0 - 1e-15))) / y_true.shape[0]
    
    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]
        self.dWs = [None] * len(self.weights)
        self.dbs = [None] * len(self.weights)
        
        dZ = y_pred - y_true
        
        for i in range(len(self.weights) - 1, -1, -1):
            self.dWs[i] = np.dot(self.A[i].T, dZ) / m
            self.dbs[i] = np.sum(dZ, axis=0, keepdims=True) / m
            
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                if self.keep_prob < 1.0 and self.D[i - 1] is not None:
                    dA = (dA * self.D[i - 1]) / self.keep_prob
                dZ = dA * self.act_deriv(self.Z[i - 1])

    def update_parameters(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.dWs[i]
            self.biases[i] -= lr * self.dbs[i]

def run_experiment(X_data, y_data, X_v, y_v, opt_mode, init_method, act_method, 
                   dropout_p, hidden_neurons, lr, batch_size=256, track_losses=False):
    
    dims = [X_data.shape[1]] + hidden_neurons + [y_data.shape[1]]
    nn = CombinatoricNeuralNetwork(
        dims=dims,
        init_method=init_method,
        activation_name=act_method,
        dropout_keep_prob=1.0 - dropout_p
    )
    
    train_losses, val_losses = [], []
    num_samples = X_data.shape[0]
    
    for epoch in range(30):
        shuffled_indices = np.random.permutation(num_samples)
        XS, yS = X_data[shuffled_indices], y_data[shuffled_indices]
        
        if opt_mode == 'batch':
            nn.backward(XS, yS, nn.forward(XS, training=True))
            nn.update_parameters(lr)
            
        elif opt_mode == 'stochastic':
            for i in range(min(num_samples, 400)):
                nn.backward(XS[i:i+1], yS[i:i+1], nn.forward(XS[i:i+1], training=True))
                nn.update_parameters(lr)
        else:
            for i in range(0, num_samples, batch_size):
                X_b, y_b = XS[i:i+batch_size], yS[i:i+batch_size]
                nn.backward(X_b, y_b, nn.forward(X_b, training=True))
                nn.update_parameters(lr)
        
        if track_losses:
            train_losses.append(nn.compute_loss(y_data, nn.forward(X_data, training=False)))
            val_losses.append(nn.compute_loss(y_v, nn.forward(X_v, training=False)))
    
    return nn, train_losses, val_losses

opt_options = ['mini_batch', 'batch']  
init_options = ['he', 'xavier']
act_options = ['relu', 'tanh']
dropout_options = [0.1, 0.2]
layer_neuron_options = [[64, 32], [128, 64], [64, 48, 32]]
lr_options = [0.01, 0.05]
batch_size_options = [128, 256]

sequential_pipeline = []
for opt in opt_options:
    for init in init_options:
        for act in act_options:
            for drop in dropout_options:
                for neurons in layer_neuron_options:
                    for lr in lr_options:
                        for bs in batch_size_options:
                            if init == 'he' and act == 'tanh':
                                continue
                            if init == 'xavier' and act == 'relu':
                                continue
                            sequential_pipeline.append({
                                'opt': opt,
                                'init': init,
                                'act': act,
                                'drop': drop,
                                'neurons': neurons,
                                'lr': lr,
                                'batch_size': bs
                            })

print(f"Всего валидных комбинаций: {len(sequential_pipeline)}")

chosen_indices = random.sample(range(len(sequential_pipeline)), min(8, len(sequential_pipeline)))

header = f"| {'№':<4} | {'Оптимизация':<12} | {'Инициализация':<14} | {'Активация':<10} | {'Drop':<5} | {'Слои':<12} | {'LR':<5} | {'BS':<4} | {'Точность':<9} |"
separator = "-" * len(header)
print(f"\n{separator}\n{header}\n{separator}")

all_results = []

for idx, config in enumerate(sequential_pipeline):
    track_losses = idx in chosen_indices
    
    print(f"\n--- Эксперимент {idx + 1}/{len(sequential_pipeline)} ---")
    
    try:
        model, train_h, val_h = run_experiment(
            X_train, y_train, X_val, y_val,
            opt_mode=config['opt'],
            init_method=config['init'],
            act_method=config['act'],
            dropout_p=config['drop'],
            hidden_neurons=config['neurons'],
            lr=config['lr'],
            batch_size=config['batch_size'],
            track_losses=track_losses
        )
        
        preds_classes = np.argmax(model.forward(X_test, training=False), axis=1)
        acc = np.mean(preds_classes == np.argmax(y_test, axis=1)) * 100
        all_results.append({'config': config, 'accuracy': acc, 'index': idx + 1})
        
        bs_str = str(config['batch_size'])
        print(f"| {idx + 1:<4} | {config['opt']:<12} | {config['init']:<14} | {config['act']:<10} | {config['drop']:<5} | {str(config['neurons']):<12} | {config['lr']:<5} | {bs_str:<4} | {acc:.2f}% |")
        
    except Exception as e:
        print(f"  ✗ Ошибка в эксперименте {idx + 1}: {e}")
        continue

print(separator)

if all_results:
    top_3 = sorted(all_results, key=lambda item: item['accuracy'], reverse=True)[:3]

    for i, res in enumerate(top_3, 1):
        c = res['config']
        print(f"{i} место: Эксперимент №{res['index']} | Точность: {res['accuracy']:.2f}% | "
              f"Опт: {c['opt']}, BS: {c['batch_size']}, Иниц: {c['init']}, Акт: {c['act']}, "
              f"Drop: {c['drop']}, Слои: {c['neurons']}, LR: {c['lr']}")

    best_model_config = top_3[0]['config']
    print(f"\nЛучшая конфигурация:")
    print(f"  Оптимизация: {best_model_config['opt']}")
    print(f"  Инициализация: {best_model_config['init']}")
    print(f"  Активация: {best_model_config['act']}")
    print(f"  Dropout: {best_model_config['drop']}")
    print(f"  Слои: {best_model_config['neurons']}")
    print(f"  Learning Rate: {best_model_config['lr']}")
    print(f"  Batch Size: {best_model_config['batch_size']}")

    best_nn, _, _ = run_experiment(
        X_train, y_train, X_val, y_val,
        opt_mode=best_model_config['opt'],
        init_method=best_model_config['init'],
        act_method=best_model_config['act'],
        dropout_p=best_model_config['drop'],
        hidden_neurons=best_model_config['neurons'],
        lr=best_model_config['lr'],
        batch_size=best_model_config['batch_size'],
        track_losses=False
    )

    y_pred_probs = best_nn.forward(X_test, training=False)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nМатрица ошибок (строки - истина, столбцы - предсказание):")
    print(cm)

    target_names = [f"Оценка {cls}" for cls in unique_classes]
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    feature_names = df.drop('quality', axis=1).columns.tolist()
    W1 = best_nn.weights[0]
    importance = np.mean(np.abs(W1), axis=1)
    sorted_idx = np.argsort(importance)[::-1]

    print("\nТоп-5 наиболее важных признаков для определения качества вина:")
    for i, idx in enumerate(sorted_idx[:5]):
        print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")

    print("\nТоп-5 наименее важных признаков:")
    for i, idx in enumerate(sorted_idx[-5:][::-1]):
        print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(unique_classes))
    plt.xticks(tick_marks, unique_classes)
    plt.yticks(tick_marks, unique_classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Предсказанная оценка')
    plt.ylabel('Истинная оценка')
    plt.title('Матрица ошибок (Wine Quality, 6500 вин)')
    plt.tight_layout()
    plt.show()

else:
    print("\nНет успешных экспериментов!")