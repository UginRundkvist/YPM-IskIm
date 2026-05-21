import numpy as np

def step_function(x):
    return 1 if x >= 0 else 0

weights_layer1_extended = np.array([
    [-0.5, 1.5],  # для 1
    [1, -1],  # Веса для x1
    [1, -1]  # Веса для x2
])

weights_layer2_extended = np.array([-1.5, 1, 1])

def two_layer(x1, x2):
    inputs = np.array([1, x1, x2])
    z_layer1 = np.dot(inputs, weights_layer1_extended)
    h_layer = np.array([step_function(z) for z in z_layer1])
    h_extended = np.array([1, h_layer[0], h_layer[1]])
    z_layer2 = np.dot(h_extended, weights_layer2_extended)
    output = step_function(z_layer2)
    return output
