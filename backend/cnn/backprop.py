import numpy as np


def dense_gradients(inputs, dL_dz):
    """
    Calculate gradients for a single Dense neuron.

    inputs:
        Values entering the neuron.

    dL_dz:
        Gradient of loss with respect to
        the neuron's weighted sum.

    Returns:
        Weight gradients and bias gradient.
    """

    inputs = np.array(inputs, dtype=float)

    dL_dweights = inputs * dL_dz

    dL_dbias = dL_dz

    return {
        "dL_dweights": dL_dweights,
        "dL_dbias": dL_dbias
    }
import numpy as np


def dense_gradients(inputs, dL_dz):
    """
    Calculate gradients for a single Dense neuron.
    """

    inputs = np.array(inputs, dtype=float)

    dL_dweights = inputs * dL_dz
    dL_dbias = dL_dz

    return {
        "dL_dweights": dL_dweights,
        "dL_dbias": dL_dbias
    }


def flatten_backward(flattened_gradient, original_shape):
    """
    Convert the flattened gradient back into
    the original feature-map shape.
    """

    flattened_gradient = np.array(
        flattened_gradient,
        dtype=float
    )

    return flattened_gradient.reshape(original_shape)