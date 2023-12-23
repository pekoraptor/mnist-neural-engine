import numpy as np
from layer import Layer
import sklearn.model_selection

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

    def fit(self, X, Y, learning_rate, gen_count, val_ratio=0.1, val_freq=float('inf')):
        if val_ratio > 0:
            X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X, Y, test_size = val_ratio)
        else:
            X_train, Y_train = X, Y
        acc = 0
        for i in range(gen_count):
            for i in np.random.permutation(len(X_train)):
                x, y = X_train[i], Y_train[i]
                output = x.reshape(len(x), 1)
                for layer in self.network:
                    output = layer.forward(output)

                y_grad = self._MSE_prime(output, y)

                for layer in reversed(self.network):
                    y_grad = layer.backward(y_grad, learning_rate)
            if val_ratio > 0:
                if i % val_freq == 0:
                    curr_acc = sum(np.argmax(pred) == np.argmax(real) for pred, real in zip(self.predict(X_val), Y_val)) / len(Y_val)
                    if curr_acc < acc:
                        break
                    acc = curr_acc


    def predict(self, X):
        ret = []
        for x in X:
            output = x.reshape(len(x), 1)
            for layer in self.network:
                output = layer.forward(output)

            ret.append(output)
        return ret
