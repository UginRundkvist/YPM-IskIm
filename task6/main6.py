import matplotlib.pyplot as plt
import numpy as np

def get_stats_standard(X):

    means = np.mean(X, axis=0)

    stds = np.std(X, axis=0, ddof=0)
    return means, stds

def normalize_method_1_max(X):

    X_max = np.max(X, axis=0)
    X_max[X_max == 0] = 1.0
    X_norm = X / X_max
    return X_norm

def normalize_method_2a_range(X, means):

    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0

    X_norm = (X - means) / X_range
    return X_norm

def normalize_method_2b_zscore(X, means, stds):

    stds[stds == 0] = 1.0
    X_norm = (X - means) / stds
    return X_norm

def plot_features(X_original, X_norm_1, X_norm_2a, X_norm_2b):

    n_features = X_original.shape[1]

    # 4 столбца (для 4 состояний) и N строк (для N признаков)
    fig, axes = plt.subplots(n_features, 4, figsize=(18, 4 * n_features))

    titles = ['Исходные данные',
              '1. ',
              '2. ',
              '3.']

    # Обеспечиваем, что axes всегда двумерный
    if n_features == 1:
        axes = np.expand_dims(axes, axis=0)

    for j in range(n_features):
        data_sets = [X_original[:, j], X_norm_1[:, j], X_norm_2a[:, j], X_norm_2b[:, j]]

        # Получаем максимальную высоту для единообразия Y-оси
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
            elif k == 1:  # Max Norm обычно в диапазоне [0, 1]
                ax.set_xlim(-0.1, 1.1)
                ax.set_xlabel('Нормированное значение')
            elif k > 1:  # Centered methods (Mean/Range и Z-score)
                ax.set_xlim(-2, 2)
                ax.set_xlabel('Нормированное значение')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    data = np.loadtxt(r"/home/zerd/all/YPM-IskIm/task6/ex1data2.txt", delimiter=',')
    print("--- Данные 'ex1data2.txt' успешно загружены ---")

    X_original = data[:, :2]

    means, stds = get_stats_standard(X_original)

    print("\n--- СТАТИСТИКА ПРИЗНАКОВ ---")
    for i in range(X_original.shape[1]):
        print(f"Признак X{i + 1}: Среднее (mu) = {means[i]:.4f}, СКО (sigma) = {stds[i]:.4f}")

    X_norm_1_max = normalize_method_1_max(X_original)
    X_norm_2a_range = normalize_method_2a_range(X_original, means)
    X_norm_2b_zscore = normalize_method_2b_zscore(X_original, means, stds)

    plot_features(X_original, X_norm_1_max, X_norm_2a_range, X_norm_2b_zscore)


# if __name__ == "__main__":
#     file_path = r"/home/zerd/all/YPM-IskIm/task6/ex1data2.txt" # C:\Users\1\Desktop\IskIn\YPM-IskIm\task6  /home/zerd/all/YPM-IskIm/task6
    
   