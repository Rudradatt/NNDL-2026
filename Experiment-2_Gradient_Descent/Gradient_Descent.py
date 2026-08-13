# Gradient Descent Implementation for Linear Regression
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5])
y = np.array([10,40,50,78,83])

m = np.random.rand(1)
c = np.zeros(1)

lr = 0.01
epochs = 10

for epoch in range(epochs):

    y_pred = m * x + c

    loss = np.mean(np.square(y_pred - y))

    error = y_pred - y
    dm = 2 * np.mean(error * x)
    dc = 2 * np.mean(error)

    m -= lr * dm
    c -= lr * dc

print('Final slope (m):', m[0])
print('Final intercept (c):', c[0])

# Graph Plotting
plt.scatter(x, y, label='Actual Data')

x_line = np.linspace(x.min()-1, x.max()+1, 100)
y_line = m * x_line + c

plt.plot(x_line, y_line, color='red', label='Regression Line')

plt.xlabel("x")
plt.ylabel("y")
plt.title("y = mx + c")
plt.legend()
plt.grid()
plt.show()