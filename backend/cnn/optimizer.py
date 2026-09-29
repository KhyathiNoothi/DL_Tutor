import numpy as np


def update_parameters(
    weights,
    biases,
    weight_gradients,
    bias_gradient,
    learning_rate
):
    weights = np.array(weights, dtype=float)
    biases = np.array(biases, dtype=float)
    weight_gradients = np.array(weight_gradients, dtype=float)

    new_weights = weights - learning_rate * weight_gradients
    new_biases = biases - learning_rate * bias_gradient

    return {
        "weights": new_weights,
        "biases": new_biases
    }