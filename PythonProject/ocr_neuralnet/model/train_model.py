import numpy as np
from tensorflow.keras.datasets import mnist
from model.neural_network import NeuralNetwork
import pickle
import os
print("Current Working Directory: ", os.getcwd())

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 784).astype('float32') / 255.0
    return X_train, y_train, X_test, y_test

def one_hot_encode(y, num_classes=10):
    encoded = np.zeros((y.size, num_classes))
    encoded[np.arange(y.size), y] = 1
    return encoded

X_train, y_train, X_test, y_test = load_mnist()
y_train_encoded = one_hot_encode(y_train)
y_test_encoded = one_hot_encode(y_test)

# Initialize Neural Network
nn = NeuralNetwork(input_size=784, hidden_layers=(128, 64), output_size=10)

# Training parameters
learning_rate = 0.01
epochs = 10
batch_size = 64

# Training Loop with Backpropagation
for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train_encoded[i:i + batch_size]

        # Forward pass
        activations = [X_batch]
        for w, b in zip(nn.weights, nn.biases):
            activations.append(nn.relu(np.dot(activations[-1], w) + b))

        # Backward pass (simple gradient calculation for output layer)
        deltas = [activations[-1] - y_batch]  # Output layer delta
        for l in range(len(nn.hidden_layers)-1, -1, -1):
            delta = np.dot(deltas[0], nn.weights[l+1].T) * (activations[l+1] > 0)  # ReLU gradient
            deltas.insert(0, delta)

        # Update weights and biases using gradient descent
        for l in range(len(nn.weights)):
            nn.weights[l] -= learning_rate * np.dot(activations[l].T, deltas[l])
            nn.biases[l] -= learning_rate * np.sum(deltas[l], axis=0, keepdims=True)

    print(f'Epoch {epoch + 1}/{epochs} complete.')

# Save trained weights
model_dir = os.path.join(os.getcwd(), 'model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'nn_weights.pkl')
with open(model_path, 'wb') as f:
    pickle.dump((nn.weights, nn.biases), f)


print("Model trained and weights saved.")
