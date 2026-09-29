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


def update_kernel(
    kernel,
    kernel_gradient,
    learning_rate
):
    kernel = np.array(kernel, dtype=float)
    kernel_gradient = np.array(kernel_gradient, dtype=float)

    if kernel.shape != kernel_gradient.shape:
        raise ValueError(
            "Kernel and kernel gradient must have the same shape"
        )

    new_kernel = kernel - learning_rate * kernel_gradient

    return new_kernel