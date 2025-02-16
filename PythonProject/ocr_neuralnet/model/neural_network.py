import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_layers=(128, 64), output_size=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        self.initialize_weights()

    def initialize_weights(self):
        layer_sizes = [self.input_size] + list(self.hidden_layers) + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(0.01 * np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def set_weights(self, weights_biases):
        self.weights, self.biases = weights_biases

    def forward(self, inputs):
        activation = inputs
        for w, b in zip(self.weights, self.biases):
            activation = self.relu(np.dot(activation, w) + b)
        return activation

    def predict(self, inputs):
        return self.forward(inputs)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)
