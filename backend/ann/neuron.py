import numpy as np


def weighted_sum(inputs, weights, bias):
    """
    Calculate the weighted sum of a neuron.

    Formula:
    z = x1*w1 + x2*w2 + ... + xn*wn + bias
    """

    inputs = np.array(inputs, dtype=float)
    weights = np.array(weights, dtype=float)

    if len(inputs) != len(weights):
        raise ValueError("Number of inputs must match number of weights")

    z = np.dot(inputs, weights) + bias

    return z