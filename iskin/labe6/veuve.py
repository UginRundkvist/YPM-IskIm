import scipy.io as sio
import numpy as np

data = sio.loadmat('C:/Users/1/Desktop/IskIn/YPM-IskIm/labe6/data.mat')
X = data['X']
y = data['y']

print(X[0])
img = X[0].reshape(20, 20)
print(img)
