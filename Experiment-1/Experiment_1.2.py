import numpy as np

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
