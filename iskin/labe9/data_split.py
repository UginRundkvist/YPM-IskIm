import scipy.io
import numpy as np

data = scipy.io.loadmat("data.mat")
x = data["X"]
y = data["y"]

permutations = np.random.permutation(x.shape[0])
x = x[permutations]
y = y[permutations]

x_test, x_train = x[:1000, :], x[1000:, :]
y_test, y_train = y[:1000, :], y[1000:, :]

scipy.io.savemat("data_test.mat", {"X": x_test, "y": y_test})
scipy.io.savemat("data_train.mat", {"X": x_train, "y": y_train})