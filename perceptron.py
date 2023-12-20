import numpy as np
import random


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
        output_value = np.dot(y_gradient, self.weights.T)
        weights_gradient = np.dot(y_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * y_gradient
        return output_value

    def _update_weights_with_sgd(self, y_gradient, learning_rate):
        weights_gradient = np.dot(y_gradient, self.weights.T)
        height, width = np.shape(weights_gradient)
        to_be_updated = []
        for i in range(height*width):
            to_be_updated.append(1 if i < height*width / 2 else 0)
        to_be_updated = np.array(random.shuffle(to_be_updated)).reshape(height, width)
        for i in range(height):
            for j in range(width):
                self.weights[i][j] -= learning_rate * weights_gradient[i][j] * to_be_updated[i][j]

    def _update_bias_with_sgd(self, y_gradient, learning_rate):
        to_be_changed = []
        for i in range(len(self.bias)):
            to_be_changed.append(1 if i < len(self.bias) / 2 else 0)
        random.shuffle(to_be_changed)
        for i in range(len(self.bias)):
            self.bias[i] -= learning_rate * y_gradient[i] * to_be_changed[i]


def relu(x):
    return x if x > 0 else 0
