#Implemented OOPs concept for architecture of single layer perceptron

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

class Perceptron:

    def __init__(self, input_size, lr = 0.1, epochs = 10) -> None:
        self.w = np.random.rand(input_size)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    def step_function(self, z) -> int:
        return 1 if z >= 0 else 0

    def forward(self, x) -> int:
        a = np.dot(x, self.w) + self.b
        return self.step_function(a)

    def fit(self, X, y) -> None:
        for epoch in range(self.epochs):
            for i in range(len(X)):
                y_pred = self.forward(X[i])

                error = y[i] - y_pred

                self.w += self.lr * error * X[i]
                self.b += self.lr * error

perceptron = Perceptron(2, 0.1, 10)
perceptron.fit(X, y)            

print("Weights:", perceptron.w)
print("Bias:", perceptron.b)  

# Decision Bundary plotting

plt.figure(figsize=(6, 6))
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color="red", s=100, label="Class 0" if i == 0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], color="green", s=100, label="Class 1" if i == 3 else "")

w = perceptron.w
b = perceptron.b

x_vals = np.array([-0.5, 1.5])

if w[1] != 0:
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, "b-", label="Decision Boundary")
else:
    plt.axvline(x=-b / w[0], color="blue", label="Decision Boundary")

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron Decision Boundary")
plt.grid(True)
plt.legend()
plt.show()