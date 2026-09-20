import numpy as np


def activation_derivative(z, activation):
    if activation == "relu":
        return np.where(z > 0, 1.0, 0.0)

    elif activation == "sigmoid":
        sigmoid_value = 1 / (1 + np.exp(-z))
        return sigmoid_value * (1 - sigmoid_value)

    elif activation == "tanh":
        tanh_value = np.tanh(z)
        return 1 - tanh_value ** 2

    elif activation == "linear":
        return np.ones_like(z)

    else:
        raise ValueError(
            "Activation must be "
            "'relu', 'sigmoid', 'tanh', or 'linear'"
        )


def calculate_layer_gradients(
    inputs,
    weights,
    biases,
    z,
    upstream_gradient,
    activation
):
    """
    Calculate gradients for all neurons in one layer.

    This is used during backpropagation.

    inputs:
        Inputs entering the layer.

    weights:
        Weight matrix of the layer.

    biases:
        Bias values of the layer.

    z:
        Weighted sums before activation.

    upstream_gradient:
        Gradient coming from the next layer.

    activation:
        Activation function used by the layer.
    """

    inputs = np.array(inputs, dtype=float)
    weights = np.array(weights, dtype=float)
    biases = np.array(biases, dtype=float)
    z = np.array(z, dtype=float)
    upstream_gradient = np.array(upstream_gradient, dtype=float)

    # Step 1: derivative of activation function
    activation_grad = activation_derivative(
        z,
        activation
    )

    # Step 2: chain rule
    dL_dz = upstream_gradient * activation_grad

    # Step 3: gradient of weights
    dL_dweights = np.outer(
        inputs,
        dL_dz
    )

    # Step 4: gradient of biases
    dL_dbiases = dL_dz

    return {
        "dL_dweights": dL_dweights.tolist(),
        "dL_dbiases": dL_dbiases.tolist(),
        "dL_dz": dL_dz.tolist()
    }