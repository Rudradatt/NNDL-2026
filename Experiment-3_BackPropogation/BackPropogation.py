# Implement Neural Network & Backpropogation Algorithm to train Neural Network on Iris Dataset

from sklearn import datasets
import numpy as np


iris = datasets.load_iris()

x = iris.data
y = (iris.target == 0).astype(float)

def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def sigmoid_derivative(b):
    return b * (1 - b)


w1 = np.random.rand(4, 2)
w2 = np.random.rand(2, 1)

lr = 0.01
epochs = 3000

for epoch in range(epochs):

    total_error = 0

    for i in range(len(x)):

        h1 = np.dot(x[i], w1)
        a1 = sigmoid(h1)

        h2 = np.dot(a1, w2)
        a2 = sigmoid(h2)

        y_pred = a2[0]
        error = y_pred - y[i]
        total_error += error ** 2

        dE_da2 = 2 * (a2 - y[i])

        da2_dh2 = sigmoid_derivative(a2)

        dE_dh2 = dE_da2 * da2_dh2

        dw2 = np.outer(a1, dE_dh2)

        dE_da1 = dE_dh2 * w2.T

        da1_dh1 = sigmoid_derivative(a1)

        dE_dh1 = dE_da1 * da1_dh1

        dw1 = np.outer(x[i], dE_dh1)


        w2 = w2 - lr * dw2

        w1 = w1 - lr * dw1


print("\nFinal W1:")
print(w1)

print("\nFinal W2:")
print(w2)