import numpy as np


class Perceptron:
    def __init__(self) -> None:
        pass

    def fit(self, X, y, learning_rate):
        pass


class Layer:
    def __init__(self, input_size, output_size) -> None:
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.input = None

    def forward(self, X):
        self.input = X
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, y_gradient, learning_rate):
        # update weights and bias using SGD and return input gradient
        pass


def relu(x):
    return x if x > 0 else 0
