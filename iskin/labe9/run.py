import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def class_to_num(n):
    if n == 10:
        return 10
    return n

class classification_forward_nn:
    def __init__(self, filename):
        self.thetas = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            self.num_layers = int(lines[0].strip())
            self.layer_sizes = list(map(int, lines[1].strip().split()))
            
            for i in range(self.num_layers - 1):
                s_in = self.layer_sizes[i]
                s_out = self.layer_sizes[i + 1]

                theta_flat = np.array(list(map(float, lines[i + 2].strip().split())))
                theta = theta_flat.reshape((s_out, s_in + 1))
                self.thetas.append(theta)

    def predict(self, x):
        a = x
        for l in range(self.num_layers - 1):
            a = np.vstack([np.ones((1, 1)), a])
            z = np.dot(self.thetas[l], a)
            a = sigmoid(z)
        return a

data = scipy.io.loadmat("data_test.mat")
x_data = data["X"]
y_data = data["y"]

permutations = np.random.permutation(x_data.shape[0])
x_data = x_data[permutations]
y_data = y_data[permutations]

nn = classification_forward_nn("weights.txt")

fig, axes = plt.subplots(3, 4, figsize=(10, 5))
fig.suptitle("Предсказания нейронной сети")

for i, ax in enumerate(axes.flatten()):
    x_input = x_data[i].reshape(-1, 1)

    res = nn.predict(x_input)
    
    prediction = np.argmax(res)
    
    res_sorted = np.sort(res[:, 0])
    if abs(res_sorted[-1] - res_sorted[-2]) < 0.1:
        prediction_str = "неясно"
    else:
        prediction_str = str(class_to_num(prediction))

    ax.imshow(x_data[i].reshape(20, 20).T, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Цифра: \"{class_to_num(y_data[i, 0] % 10)}\"\nПредсказание: \"{prediction_str}\"")
    ax.axis("off")
    
plt.tight_layout()
plt.show()