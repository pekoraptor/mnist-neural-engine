import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.model_selection
from perceptron import Perceptron
from activations import sigmoid, sigmoid_prime
from copy import copy

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

def format_perceptron_output(Y):
    output = []
    for result in Y:
        output.append(np.argmax(result))

    return output

def check_accuracy(Y_predicted, Y_real):
    return sum(np.argmax(pred) == np.argmax(real) for pred, real in zip(Y_predicted, Y_real)) / len(Y_predicted) * 100

def analyze_learning_rate(X, Y, perceptron, learning_rate_arr, gen_count=100, t_size=0.5, sample_count=10):
    output = []
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = t_size)
    for learning_rate in learning_rate_arr:
        sample = []
        for _ in range(sample_count):
            p = copy(perceptron)
            p.fit(X_train, Y_train, learning_rate, gen_count, val_freq=100)
            sample.append(check_accuracy(p.predict(X_test), Y_test))
        output.append(copy(sample))
    return output

def plt_accuracy(learning_rate_arr, acc_arr):
    x = [str(lr) for lr in learning_rate_arr]
    y = [sum(acc)/len(acc) for acc in acc_arr]
    y_err = [np.std(acc) for acc in acc_arr]

    plt.bar(x, y, yerr=y_err, capsize=5, color='slategray')
    plt.title("Comparison of different learning rates")
    plt.xlabel("Learning rate")
    plt.ylabel("Accuracy [%]")
    for i, acc in enumerate(y):
        plt.text(i, 0.1, f'{acc:.2f}%', ha='center', va='bottom', color='black')

    plt.show()



if __name__ == "__main__":
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2)
    # Y = np.array([0, 1, 1, 0]).reshape(4, 1)
    # perceptron = Perceptron(2, 3, 1, 2, sigmoid, sigmoid_prime)
    perceptron = Perceptron(64, 10, 3, 10, sigmoid, sigmoid_prime)
    X, Y  = process_dataset()
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.5, random_state=42)
    perceptron.fit(X_train, Y_train, 0.09, 500, val_freq=10000)
    predict_output = format_perceptron_output(perceptron.predict(X_test))
    Y_test = format_perceptron_output(Y_test)
    for y_pred, y_real in zip(predict_output, Y_test):
        print(f'{y_pred}, {y_real}')
    print(f'Accuracy: {check_accuracy(predict_output, Y_test)}%')
    # plt_function3D(g, sgd(g, gGradient, [1, 1.7], 0.5, 10000, 0.01, 4, 1)[1], ['red', 'blue', 'green'])


def XOR_test():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2)
    Y = np.array([0, 1, 1, 0]).reshape(4, 1)

    perceptron = Perceptron(2, 3, 1, 2, sigmoid, sigmoid_prime)
    perceptron.fit(X, Y, 1, 1000, 0)
    output = perceptron.predict(X)

    data = {}
    data["Input"] = list(X)
    data["Predicted"] = list(output)
    data["Correct Output"] = list(Y)

    return pd.DataFrame(data, index=list(range(len(output))))