import numpy as np

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
        y_grad_after_act = self._activation_backward(y_gradient)

        output_value = np.dot(self.weights.T, y_grad_after_act)
        self._dense_backward(y_grad_after_act, learning_rate)
        return output_value

    def _dense_backward(self, y_gradient, learning_rate):
        weights_gradient = np.dot(y_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * y_gradient

    def _activation_backward(self, y_gradient):
        return np.multiply(y_gradient, self.activation_prime(self.act_input))
