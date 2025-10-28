import numpy as np
import matplotlib.pyplot as plt
loss_history = []

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = 8000

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

for epoch in range(epochs):
    s1 = np.dot(X, W1) + b1
    x1 = sigmoid(s1)
    s2 = np.dot(x1, W2) + b2
    x2 = sigmoid(s2)
    
    loss = np.mean((y - x2) ** 2)
    loss_history.append(loss)
    
    d_x2 = x2 - y
    d_s2 = d_x2 * sigmoid_derivative(x2)
    d_W2 = np.dot(x1.T, d_s2)
    d_b2 = np.sum(d_s2, axis=0, keepdims=True)

    d_x1 = np.dot(d_s2, W2.T)
    d_s1 = d_x1 * sigmoid_derivative(x1)
    d_W1 = np.dot(X.T, d_s1)
    d_b1 = np.sum(d_s1, axis=0, keepdims=True)
    
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    
print("\nFinal Predictions after training:")
for i, x in enumerate(X):
    s1 = np.dot(x, W1) + b1
    x1 = sigmoid(s1)
    s2 = np.dot(x1, W2) + b2
    x2 = sigmoid(s2)
    print(f"Input: {x} -> Predicted: {float(x2[0]):.4f} | Expected: {y[i][0]}")
    
plt.figure(figsize=(8, 5))
plt.plot(loss_history, color='blue')
plt.title("Loss Curve for XOR Training")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.show()
