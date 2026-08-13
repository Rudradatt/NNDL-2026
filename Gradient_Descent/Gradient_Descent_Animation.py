# Gradient Descent Implementation for Linear Regression and showing regression line animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 40, 50, 78, 83])

m = np.random.rand()
c = 0.0

lr = 0.01
epochs = 100

m_history = []
c_history = []

for epoch in range(epochs):

    y_pred = m * x + c

    loss = np.mean((y_pred - y) ** 2)

    error = y_pred - y

    dm = 2 * np.mean(error * x)
    dc = 2 * np.mean(error)

    m -= lr * dm
    c -= lr * dc

    m_history.append(m)
    c_history.append(c)

fig, ax = plt.subplots()

ax.scatter(x, y, color='blue', label='Data')

line, = ax.plot([], [], color='red', label='Regression Line')

ax.set_xlim(0, 6)
ax.set_ylim(0, 100)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)


def update(frame):

    current_m = m_history[frame]
    current_c = c_history[frame]

    x_line = np.linspace(0, 6, 100)
    y_line = current_m * x_line + current_c

    line.set_data(x_line, y_line)

    ax.set_title(
        f"Epoch: {frame + 1} | "
        f"Slope (m): {current_m:.2f} | "
        f"Intercept (c): {current_c:.2f}"
    )

    return line,

ani = FuncAnimation(fig, update, frames=len(m_history), interval=100, repeat=False)
plt.show()