import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os

LAYER_SIZES = [400, 25, 10]

ALPHA = 1.5
LAMBDA_REG = 1.0
STEPS = 2000
TEST_SIZE = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    a = sigmoid(z)
    return a * (1 - a)

def class_to_num(n):
    if n == 10:
        return 10
    return n

class ClassificationForwardNN:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.thetas = []

        epsilon_init = 0.12

        for i in range(self.num_layers - 1):
            s_in = layer_sizes[i]
            s_out = layer_sizes[i + 1]

            theta = (
                np.random.rand(s_out, s_in + 1)
                * 2
                * epsilon_init
                - epsilon_init
            )

            self.thetas.append(theta)

    def compute_cost_and_gradients(self, x, y_per_class, lambda_reg):
        m = x.shape[0]

        a_cache = []
        z_cache = []

        a = x.T
        for l in range(self.num_layers - 1):
            a = np.vstack([np.ones((1, m)), a])

            a_cache.append(a)

            z = np.dot(self.thetas[l], a)

            z_cache.append(z)

            a = sigmoid(z)

        a_cache.append(a)

        h = a
        y = y_per_class.T

        cost = np.sum(
            -y * np.log(h + 1e-15)
            - (1 - y) * np.log(1 - h + 1e-15)
        ) / m

        reg = 0

        for theta in self.thetas:
            reg += np.sum(theta[:, 1:] ** 2)

        cost += (lambda_reg / (2 * m)) * reg

        deltas = [0] * self.num_layers

        deltas[-1] = h - y

        for l in range(self.num_layers - 2, 0, -1):
            theta = self.thetas[l]
            theta_no_bias = theta[:, 1:]

            z = z_cache[l - 1]

            deltas[l] = (
                np.dot(theta_no_bias.T, deltas[l + 1]) * sigmoid_gradient(z)
            )

        gradients = []

        for l in range(self.num_layers - 1):
            grad = (
                np.dot(deltas[l + 1], a_cache[l].T)
                / m
            )

            reg_grad = (lambda_reg / m) * self.thetas[l]
            reg_grad[:, 0] = 0

            gradients.append(grad + reg_grad)

        return cost, gradients

    def train(self, x, y, alpha, lambda_reg, steps):
        m = x.shape[0]

        y_per_class = np.zeros((m, self.layer_sizes[-1]))

        for i in range(m):
            y_per_class[i, y[i, 0] % 10] = 1

        print("Градиентный спуск начат")
        print("------------------------------------")
        print(" Шаг   | Значение функции стоимости ")
        print("------------------------------------")

        for step in range(steps):
            cost, gradients = self.compute_cost_and_gradients(
                x,
                y_per_class,
                lambda_reg
            )

            for l in range(len(self.thetas)):
                self.thetas[l] -= alpha * gradients[l]

            if step % 250 == 0:
                print(f" {step:<6}| {cost:.4f}")


    def predict(self, x):
        a = x

        for l in range(self.num_layers - 1):
            a = np.vstack([np.ones((1, 1)), a])

            z = np.dot(self.thetas[l], a)

            a = sigmoid(z)

        return a

    def save_weights(self, filename):
        with open(filename, "w") as f:
            f.write(f"{len(self.layer_sizes)}\n")
            f.write(" ".join(map(str, self.layer_sizes)) + "\n")

            for theta in self.thetas:
                np.savetxt(f, theta.flatten(), newline=" ")
                f.write("\n")
                
    def load_weights(self, filename):
        self.thetas = []

        with open(filename, "r") as f:
            lines = f.readlines()

            self.num_layers = int(lines[0].strip())
            self.layer_sizes = list(map(int, lines[1].strip().split()))

            for i in range(self.num_layers - 1):
                s_in = self.layer_sizes[i]
                s_out = self.layer_sizes[i + 1]

                theta_flat = np.array(
                    list(map(float, lines[i + 2].strip().split()))
                )

                theta = theta_flat.reshape((s_out, s_in + 1))

                self.thetas.append(theta)          



data = scipy.io.loadmat("C:/Users/1/Desktop/IskIn/YPM-IskIm/iskin/labe9/data.mat")

x = data["X"]
y = data["y"]

print("Размер исходного набора данных:")
print(x.shape)

permutations = np.random.permutation(x.shape[0])

x = x[permutations]
y = y[permutations]


x_test = x[:TEST_SIZE]
y_test = y[:TEST_SIZE]

x_train = x[TEST_SIZE:]
y_train = y[TEST_SIZE:]

print("\nОбучающая выборка:")
print(x_train.shape)

print("\nТестовая выборка:")
print(x_test.shape)

WEIGHTS_FILE = "weights.txt"

nn = ClassificationForwardNN(LAYER_SIZES)

if os.path.exists(WEIGHTS_FILE):
    print(f"\nНайден файл {WEIGHTS_FILE}")
    print("Загрузка сохранённых весов...")

    nn.load_weights(WEIGHTS_FILE)

    print("Веса успешно загружены")

else:
    print("\nФайл весов не найден")
    print("Начинается обучение сети...")

    nn.train(
        x_train,
        y_train,
        ALPHA,
        LAMBDA_REG,
        STEPS
    )

    nn.save_weights(WEIGHTS_FILE)

    print(f"\nВеса сохранены в {WEIGHTS_FILE}")
    print("Обучение завершено")
    
correct = 0

for i in range(len(x_test)):
    x_input = x_test[i].reshape(-1, 1)

    result = nn.predict(x_input)

    prediction = np.argmax(result)

    real_digit = y_test[i, 0] % 10

    if prediction == real_digit:
        correct += 1

accuracy = correct / len(x_test) * 100

print("\n======================")
print("РЕЗУЛЬТАТ ТЕСТИРОВАНИЯ")
print("======================")

print(f"Правильных ответов: {correct} из {len(x_test)}")
print(f"Точность сети: {accuracy:.2f}%")

fig, axes = plt.subplots(3, 4, figsize=(10, 5))
fig.suptitle("Предсказания нейронной сети")

for i, ax in enumerate(axes.flatten()):
    x_input = x_test[i].reshape(-1, 1)

    result = nn.predict(x_input)

    prediction = np.argmax(result)

    sorted_result = np.sort(result[:, 0])

    if abs(sorted_result[-1] - sorted_result[-2]) < 0.1:
        prediction_str = "неясно"
    else:
        prediction_str = str(class_to_num(prediction))

    ax.imshow(
        x_test[i].reshape(20, 20).T,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    real_digit = class_to_num(y_test[i, 0] % 10)

    ax.set_title(
        f'Цифра: "{real_digit}"\n'
        f'Предсказание: "{prediction_str}"'
    )

    ax.axis("off")

plt.tight_layout()
plt.show()