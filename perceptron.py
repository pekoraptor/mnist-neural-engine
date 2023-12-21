import numpy as np
import random
import math


class Perceptron:
    def __init__(self, input_size, n_count, layer_count, class_count, activation_function, activation_prime) -> None:
        self.network = []
        # initialize neural network
        self.network.append(Layer(input_size, n_count, activation_function, activation_prime))
        for _ in range(layer_count - 1):
            self.network.append(Layer(n_count, n_count, activation_function, activation_prime))
        # for binary classification output layer has 1 neuron. otherwise class_count neurons
        self.network.append(Layer(n_count, class_count if class_count != 2 else 1, activation_function, activation_prime))
        self.activation_function = activation_function
        self.activation_prime = activation_prime

    def _MSE(self, predicted_Y, Y):
        return sum(np.power(predicted_Y[i] - Y[i], 2) for i in range(len(Y)))

    def _MSE_prime(self, predicted_Y, Y):
        return 2 * (predicted_Y - Y) / np.size(Y)

    def fit(self, X, y, learning_rate, gen_count):
        for i in range(gen_count):
            for x, y in zip(X, Y):
                for layer in self.network:
                    x = layer.forward(x)

                y_grad = self._MSE_prime(y, x)
                
                for layer in reversed(self.network):
                    layer.backward(y_grad, learning_rate)

    def predict(self, X):
        pass


class Layer:
    def __init__(self, input_size, output_size, activation_function, activation_prime) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.input = None
        self.act_input = None
        self.activation_function = activation_function
        self.activation_prime = activation_prime
    
    def __repr__(self):
        return repr(f"Layer: {self.input_size},  {self.output_size}")

    def forward(self, X):
        self.input = X
        return self._activation_forward(self._dense_forward())

    def _dense_forward(self):
        return np.dot(self.weights, self.input) + self.bias
    
    def _activation_forward(self, X):
        self.act_input = X
        return self.activation_function(X)

    def backward(self, y_gradient, learning_rate):
        # update weights and bias
        output_value = np.dot(y_gradient, self.weights.T)
        self._dense_backward(self._activation_backward(y_gradient), learning_rate)
        return output_value
    
    def _dense_backward(self, y_gradient, learning_rate):
        weights_gradient = np.dot(y_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * y_gradient

    def _activation_backward(self, y_gradient):
        return np.multiply(y_gradient, self.activation_prime(self.act_input))

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


def softmax(X):
    s = sum(np.exp(X))
    return (np.exp(X) / s).reshape(X.shape)

def softmax_prime(X):
    s = softmax(X)
    return (s * (1 - s)).reshape(X.shape)


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2)
    Y = np.array([0, 1, 1, 0]).reshape(4, 1)
    perceptron = Perceptron(2, 3, 1, 2, softmax, softmax_prime)
    perceptron.fit(X, Y, 0.01, 1000)