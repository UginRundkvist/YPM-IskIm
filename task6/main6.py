import numpy as np
import matplotlib.pyplot as plt
import os



def get_stats_standard(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=0) # стандартное отклоненеи
    return means, stds

#три метода нормировки
def normalize_method_1(X): # x_j = x_j / x_{j max}
    X_max = np.max(X, axis=0)
    X_max[X_max == 0] = 1.0
    X_norm = X / X_max
    return X_norm


def normalize_method_2a(X, means): #(x_j - mu_j) / (x_{j max} - x_{j min})
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0

    X_norm = (X - means) / X_range
    return X_norm


def normalize_method_2b(X, means, stds): #(x_j - mu_j) / sigma_j (Z-score)
    stds[stds == 0] = 1.0
    X_norm = (X - means) / stds
    return X_norm


def plot_features(X_original, X_norm_1, X_norm_2a, X_norm_2b):
    n_features = X_original.shape[1]

    fig, axes = plt.subplots(n_features, 4, figsize=(18, 4 * n_features))

    titles = ['Исходные данные',
              '1. Деление на максимум',
              '2. (x - μ) / (max-min)',
              '3. Z-score (x - μ)/σ']

    if n_features == 1:
        axes = np.expand_dims(axes, axis=0)

    for j in range(n_features):
        data_sets = [X_original[:, j], X_norm_1[:, j], X_norm_2a[:, j], X_norm_2b[:, j]]

        counts, _ = np.histogram(data_sets[0], bins=15)
        y_max = np.max(counts) * 1.1

        for k in range(4):  # Итерация по 4 методам
            ax = axes[j, k]
            ax.hist(data_sets[k], bins=15, edgecolor='black', alpha=0.7)
            ax.set_title(f'{titles[k]} - X{j + 1}')

            ax.set_ylim(0, y_max)

            # Настройка оси X для нормированных графиков
            if k == 0:
                ax.set_xlabel(f'X{j + 1} (Original)')
            elif k == 1:  
                ax.set_xlim(-0.1, 1.1)
                ax.set_xlabel('Нормированное значение')
            elif k > 1: 
                ax.set_xlim(-2, 2)
                ax.set_xlabel('Нормированное значение')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = "/home/zerd/all/YPM-IskIm/task6/ex1data2.txt"
    
    try:
        data = np.loadtxt(file_path, delimiter=',')
        print(f"--- Данные '{os.path.basename(file_path)}' успешно загружены ---")
#указание на основную программу
        print(f"Размер данных: {data.shape}")
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        exit(1)

    # Извлекаем признаки X1 и X2 
    X_original = data[:, :2]

    means, stds = get_stats_standard(X_original)
 
    X_norm_1_max = normalize_method_1(X_original)
    X_norm_2a_range = normalize_method_2a(X_original, means)
    X_norm_2b_zscore = normalize_method_2b(X_original, means, stds)

 
    plot_features(X_original, X_norm_1_max, X_norm_2a_range, X_norm_2b_zscore)