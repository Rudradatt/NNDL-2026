import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

w1 = np.random.rand(1)
w2 = np.random.rand(1)
b = np.zeros(1)

lr = 0.1
epochs = 10

def step_function(z):
  if z > 0:
    return 1
  else:
    return 0
  
for epoch in range (epochs):
  for i in range (len(x)):
    a = w1 * x[i][0] + w2 * x[i][1] + b
    y_pred = step_function(a)

    error = y[i] - y_pred

    w1 += lr * error * x[i][0]
    w2 += lr * error * x[i][1]
    b += lr * error

print('Weight 1:',w1)
print('Weight 2:',w2)
print('bias',b)

  