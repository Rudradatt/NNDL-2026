# Network:  image(28x28) -> Conv(4 filters, 3x3) -> ReLU -> MaxPool(2x2) -> Flatten -> Dense(10) -> Softmax
import numpy as np
np.random.seed(0)

# import os, urllib.request
# if not os.path.exists("mnist.npz"):                     # download only the first time
#     print("Downloading MNIST...")
#     urllib.request.urlretrieve(
#         "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", "mnist.npz")
data = np.load("mnist.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_test,  y_test  = data["x_test"],  data["y_test"]
x_train = x_train[:1000] / 255.0 
y_train = y_train[:1000]
x_test  = x_test[:200] / 255.0
y_test  = y_test[:200]

NF = 4                                        # number of filters
W1 = np.random.randn(NF, 3, 3) * 0.1          # 4 filters of each 3x3
b1 = np.zeros(NF)
W2 = np.random.randn(NF * 13 * 13, 10) * 0.1  # dense layer of 676 inputs that gives 10 outputs
b2 = np.zeros(10)

def forward(img):
    conv = np.zeros((NF, 26, 26))
    for f in range(NF):
        for i in range(26):
            for j in range(26):
                patch = img[i:i+3, j:j+3]
                conv[f, i, j] = np.sum(patch * W1[f]) + b1[f]

    relu = np.maximum(0, conv)

    pool = np.zeros((NF, 13, 13))
    for f in range(NF):
        for i in range(13):
            for j in range(13):
                pool[f, i, j] = np.max(relu[f, 2*i:2*i+2, 2*j:2*j+2])

    flat = pool.flatten()
    scores = flat @ W2 + b2

    e = np.exp(scores - np.max(scores))
    probs = e / np.sum(e)
    return conv, relu, pool, flat, probs

def backward(img, label, conv, relu, pool, flat, probs):
    d_scores = probs.copy()
    d_scores[label] -= 1

    dW2 = np.outer(flat, d_scores)
    db2 = d_scores
    d_flat = W2 @ d_scores

    d_pool = d_flat.reshape(NF, 13, 13)

    d_relu = np.zeros((NF, 26, 26))
    for f in range(NF):
        for i in range(13):
            for j in range(13):
                block = relu[f, 2*i:2*i+2, 2*j:2*j+2]
                r, c = np.unravel_index(np.argmax(block), block.shape)
                d_relu[f, 2*i + r, 2*j + c] = d_pool[f, i, j]

    d_conv = d_relu * (conv > 0)

    dW1 = np.zeros_like(W1)
    db1 = np.zeros(NF)
    for f in range(NF):
        for i in range(26):
            for j in range(26):
                dW1[f] += d_conv[f, i, j] * img[i:i+3, j:j+3]
                db1[f] += d_conv[f, i, j]
    return dW1, db1, dW2, db2

lr = 0.01
for epoch in range(3):
    total_loss = 0
    for n in range(len(x_train)):
        img, label = x_train[n], y_train[n]

        conv, relu, pool, flat, probs = forward(img)          # forward
        total_loss += -np.log(probs[label] + 1e-9)            # loss
        dW1, db1, dW2, db2 = backward(img, label, conv, relu, pool, flat, probs)  # backward

        W1 -= lr * dW1;  b1 -= lr * db1                       # update weights
        W2 -= lr * dW2;  b2 -= lr * db2

    correct = sum(np.argmax(forward(x_test[k])[-1]) == y_test[k] for k in range(len(x_test)))
    print(f"Epoch {epoch+1}: loss = {total_loss/len(x_train):.3f}, test accuracy = {correct/len(x_test)*100:.1f}%")

np.set_printoptions(precision=3, suppress=True)
# print("Updated Weights")
# for f in range(NF):
#     print(f"\nFilter {f+1} (3x3):")
#     print(W1[f])
#     print(f"Bias: {b1[f]:.3f}")
# print("\nDense layer weights shape:", W2.shape, "(676 inputs -> 10 digits)")
# print("First 3 rows of dense weights:")
# print(W2[:3])
# print("Dense biases (one per digit 0-9):")
# print(b2)

print("classifying test images")
print("Image  Actual  Predicted  Confidence  Result")
for k in range(10):
    probs = forward(x_test[k])[-1]
    pred = np.argmax(probs)
    result = "correct" if pred == y_test[k] else "wrong"
    print(f"{k:5d}  {y_test[k]:6d}  {pred:9d}  {probs[pred]*100:9.1f}%  {result}")

print("Probabilities for test image 0 (actual digit = %d):" % y_test[0])
probs = forward(x_test[0])[-1]
for d in range(10):
    print(f"  Digit {d}: {probs[d]*100:5.1f}%  " + "" * int(probs[d] * 30))

print("Accuracy per digit (on %d test images)" % len(x_test))
preds = np.array([np.argmax(forward(x_test[k])[-1]) for k in range(len(x_test))])
for d in range(10):
    idx = (y_test == d)
    if idx.sum() > 0:
        print(f"Digit {d}: {np.mean(preds[idx] == d)*100:5.1f}%  ({idx.sum()} images)")
print(f"Overall accuracy: {np.mean(preds == y_test)*100:.1f}%")