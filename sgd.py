from copy import copy
import random
import math


def sgd(f, f_gradient, x0, learning_rate, max_depth, precision, modulation_rate, batch_size):
    valueArray = []
    depth = 0
    next_x = copy(x0)

    while depth <= max_depth:
        valueArray.append(copy(next_x))
        depth += 1
        current_x = copy(next_x)

        indexes = random.sample(list(range(len(x0))), batch_size)

        f_gradient_value = []
        for index in indexes:
            f_gradient_value.append(f_gradient(current_x, index))
            next_x[index] = (current_x[index]
                            - learning_rate * f_gradient_value[-1])

        if (sum(abs(value) for value in f_gradient_value) <= precision
            or all(abs(next_x[index] - current_x[index]) <= precision
                    for index in range(len(x0)))):
            return next_x, valueArray

        if (f(next_x) >= f(current_x)):
            learning_rate /= modulation_rate
    return next_x, valueArray

