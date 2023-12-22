import numpy as np

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
        return np.mean(np.power(Y - predicted_Y, 2))

    def _MSE_prime(self, predicted_Y, Y):
        return 2 * (predicted_Y - Y) / np.size(Y)

    def fit(self, X, Y, learning_rate, gen_count):
        for _ in range(gen_count):
            for i in np.random.permutation(len(X)):
                x, y = X[i], Y[i]
                output = x.reshape(len(x), 1)
                for layer in self.network:
                    output = layer.forward(output)

                y_grad = self._MSE_prime(output, y)

                for layer in reversed(self.network):
                    y_grad = layer.backward(y_grad, learning_rate)


    def predict(self, X):
        ret = []
        for x in X:
            output = x.reshape(len(x), 1)
            for layer in self.network:
                output = layer.forward(output)

            ret.append(output)
        return ret


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
        # self._update_weights_with_sgd(y_gradient, learning_rate)
        # self._update_bias_with_sgd(y_gradient, learning_rate)

    def _activation_backward(self, y_gradient):
        return np.multiply(y_gradient, self.activation_prime(self.act_input))

    # def _update_weights_with_sgd(self, y_gradient, learning_rate):
    #     weights_gradient = np.dot(self.weights.T, y_gradient)
    #     height, width = np.shape(self.weights)
    #     to_be_updated = []
    #     for i in range(height*width):
    #         to_be_updated.append(1 if i < height*width / 2 else 0)
    #     random.shuffle(to_be_updated)
    #     to_be_updated = np.array(to_be_updated).reshape(height, width)
    #     self.weights -= learning_rate * np.multiply(weights_gradient.T, to_be_updated)

    # def _update_bias_with_sgd(self, y_gradient, learning_rate):
    #     to_be_changed = []
    #     for i in range(len(self.bias)):
    #         to_be_changed.append(1 if i < len(self.bias) / 2 else 0)
    #     random.shuffle(to_be_changed)
    #     for i in range(len(self.bias)):
    #         self.bias[i] -= learning_rate * y_gradient[i] * to_be_changed[i]


