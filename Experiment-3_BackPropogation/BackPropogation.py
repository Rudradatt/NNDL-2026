from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()

x = iris.data


y = (iris.target == 0).astype(float)



def sigmoid(a):
    return 1 / (1 + np.exp(-a))



w1 = np.random.rand(4, 2)
w2 = np.random.rand(2, 1)

lr = 0.01
epochs = 3000

error_history = []

for epoch in range(epochs):

    total_error = 0

    for i in range(len(x)):


        h1 = np.dot(x[i], w1)

        a1 = sigmoid(h1)

        # Calculate derivative once
        da1_dh1 = a1 * (1 - a1)


  
        h2 = np.dot(a1, w2)

        a2 = sigmoid(h2)

        da2_dh2 = a2 * (1 - a2)

        y_pred = a2[0]


        error = y_pred - y[i]

        total_error += np.square(error)


        # Error with respect to output
        dE_da2 = 2 * (a2 - y[i])

        # Error with respect to h2
        dE_dh2 = dE_da2 * da2_dh2


        # Gradient of w2
        dw2 = np.outer(a1, dE_dh2)


        # Error propagated to hidden layer
        dE_da1 = dE_dh2 * w2.T

        # Error with respect to h1
        dE_dh1 = dE_da1 * da1_dh1


        # Gradient of w1
        dw1 = np.outer(x[i], dE_dh1)



        w2 = w2 - lr * dw2

        w1 = w1 - lr * dw1


    # Store error
    error_history.append(total_error)


# -------------------------
# Final Weights
# -------------------------

print("\nFinal W1:")
print(w1)

print("\nFinal W2:")
print(w2)


# -------------------------
# Prediction
# -------------------------

print("\n-------------------------")
print("Predictions")
print("-------------------------")

correct = 0

for i in range(len(x)):

    # Forward Pass

    h1 = np.dot(x[i], w1)

    a1 = sigmoid(h1)

    h2 = np.dot(a1, w2)

    a2 = sigmoid(h2)

    y_pred = a2[0]


    # Convert probability to class

    if y_pred >= 0.5:
        predicted_class = 1
    else:
        predicted_class = 0


    # Actual and predicted names

    actual_name = "Setosa" if y[i] == 1 else "Not Setosa"

    predicted_name = "Setosa" if predicted_class == 1 else "Not Setosa"


    # Print prediction

    print(
        "Actual:",
        actual_name,
        "| Predicted:",
        predicted_name,
        "| Probability:",
        round(y_pred, 4)
    )


    # Check prediction

    if predicted_class == y[i]:
        correct += 1


# -------------------------
# Calculate Accuracy
# -------------------------

accuracy = (correct / len(x)) * 100

print("\n-------------------------")
print("Accuracy")
print("-------------------------")

print("Correct Predictions:", correct)

print("Total Predictions:", len(x))

print("Accuracy:", accuracy, "%")


plt.figure(figsize=(8, 5))

plt.plot(
    range(epochs),
    error_history,
    linewidth=1
)

plt.title("Training Error vs Epochs")

plt.xlabel("Epoch")

plt.ylabel("Total Squared Error")

plt.grid(True)

plt.show()