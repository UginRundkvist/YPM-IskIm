import numpy as np
import scipy.io

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    a = sigmoid(z)
    return a * (1 - a)

class classification_forward_nn:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.thetas = []
         
        epsilon_init = 0.12
        for i in range(self.num_layers - 1):
            s_in = layer_sizes[i]
            s_out = layer_sizes[i+1]
            theta = np.random.rand(s_out, s_in + 1) * 2 * epsilon_init - epsilon_init
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
        cost = np.sum(-y * np.log(h + 1e-15) - (1 - y) * np.log(1 - h + 1e-15)) / m
        
        reg = 0
        for theta in self.thetas:
            reg += np.sum(theta[:, 1:] ** 2)
        cost += (lambda_reg / (2 * m)) * reg
        
        deltas = [0] * self.num_layers
        deltas[-1] = h - y
        
        for l in range(self.num_layers - 2, 0, -1):
            theta = self.thetas[l]

            theta_no_bias = theta[:, 1:] 
            z = z_cache[l-1]
            
            delta = np.dot(theta_no_bias.T, deltas[l+1]) * sigmoid_gradient(z)
            deltas[l] = delta

        gradients = []
        for l in range(self.num_layers - 1):
            d_matrix = np.dot(deltas[l+1], a_cache[l].T) / m
            
            reg_grad = (lambda_reg / m) * self.thetas[l]
            reg_grad[:, 0] = 0
            
            gradients.append(d_matrix + reg_grad)
            
        return cost, gradients

    def train(self, x, y, alpha, lambda_reg, steps):
        m = x.shape[0]
        y_per_class = np.zeros((m, layer_sizes[-1]))
        for i in range(m):
            y_per_class[i, y[i, 0] % 10] = 1
            
        print("Градиентный спуск начат")
        print("------------------------------------")
        print(" Шаг   | Значение функции стоимости ")
        print("------------------------------------")
        for step in range(steps):
            cost, gradients = self.compute_cost_and_gradients(x, y_per_class, lambda_reg)
            
            for l in range(len(self.thetas)):
                self.thetas[l] -= alpha * gradients[l]
                
            if step % 250 == 0:
                print(f" {step:<6}| {cost:.4f}")
        print("------------------------------------")
                
    def save_weights(self, filename):
        with open(filename, 'w') as f:
            f.write(f"{len(self.layer_sizes)}\n")
            f.write(" ".join(map(str, self.layer_sizes)) + "\n")
            for theta in self.thetas:
                np.savetxt(f, theta.flatten(), newline=" ")
                f.write("\n")

data = scipy.io.loadmat("data_train.mat")
x = data["X"]
y = data["y"]

num_layers = int(input("Введите общее количество слоев: "))
layer_sizes = []
for i in range(num_layers):
    size = int(input(f"Введите количество нейронов для слоя {i + 1}: "))
    layer_sizes.append(size)

nn = classification_forward_nn(layer_sizes)

alpha = 1.5
lambda_reg = 1.0
steps = 2000

nn.train(x, y, alpha, lambda_reg, steps)
nn.save_weights("weights.txt")
print("Веса сохранены в weights.txt")