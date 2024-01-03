import numpy as np


def sigmoid(x):
    limited_x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-limited_x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
