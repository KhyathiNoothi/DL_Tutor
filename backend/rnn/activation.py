import numpy as np


def tanh(x):
    x = np.array(x, dtype=float)
    return np.tanh(x)


def tanh_derivative(x):
    x = np.array(x, dtype=float)

    tanh_value = np.tanh(x)

    return 1 - tanh_value ** 2


def sigmoid(x):
    x = np.array(x, dtype=float)

    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sigmoid_value = sigmoid(x)

    return sigmoid_value * (1 - sigmoid_value)