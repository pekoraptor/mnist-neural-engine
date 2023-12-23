import numpy as np

def softmax(X):
    s = sum(np.exp(X))
    return (np.exp(X) / s).reshape(X.shape)

def softmax_prime(X):
    s = softmax(X)
    return (s * (1 - s)).reshape(X.shape)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
