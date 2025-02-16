import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize and flatten
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

    return X_train, y_train, X_test, y_test
