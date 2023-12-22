import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.model_selection
from perceptron import Perceptron

def plt_function3D(f, additionalPoints=None, pointsColors=None,
                   pltCmap='PiYG', visibility=0.3):
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    x1, x2 = np.meshgrid(x1, x2)
    y = np.zeros_like(x1)

    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            y[i, j] = f([x1[i, j], x2[i, j]])

    figure = plt.figure()
    plot = figure.add_subplot(111, projection='3d')
    plot.plot_surface(x1, x2, y, cmap=pltCmap, alpha=visibility)

    if additionalPoints:
        for index, setOfPoints in enumerate(additionalPoints):
            additionalX1, additionalX2 = setOfPoints
            plot.scatter(additionalX1, additionalX2, f(setOfPoints),
                         c=pointsColors[index % len(pointsColors)],
                         s=100)

    plot.set_xlabel('x1')
    plot.set_ylabel('x2')
    plot.set_zlabel('g(x1, x2)')
    plot.set_title('3D Plot of g(x1, x2)')

    plt.legend()
    plt.show()

def print_mnist():
    dataset = sklearn.datasets.load_digits()
    for line in dataset['images'][12]:
        print([int(x > 0) for x in line])


def process_dataset():
    dataset = sklearn.datasets.load_digits()
    X = dataset['data']
    Y = dataset['target']
    Y_formatted = []

    for y in Y:
        formatted = [0] * y + [1] + [0] * (9 - y)
        Y_formatted.append(np.array(formatted).reshape(len(formatted), 1))

    return X, Y_formatted

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

if __name__ == "__main__":
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2)
    # Y = np.array([0, 1, 1, 0]).reshape(4, 1)
    # perceptron = Perceptron(2, 3, 1, 2, sigmoid, sigmoid_prime)
    perceptron = Perceptron(64, 10, 3, 10, sigmoid, sigmoid_prime)
    X, Y  = process_dataset()
    perceptron.fit(X, Y, 0.01, 1000)
    print(perceptron.predict(np.array([X[15]])))
    print(Y[15])
    # plt_function3D(g, sgd(g, gGradient, [1, 1.7], 0.5, 10000, 0.01, 4, 1)[1], ['red', 'blue', 'green'])