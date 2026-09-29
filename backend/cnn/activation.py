import numpy as np


def relu(feature_map):
    feature_map = np.array(feature_map, dtype=float)
    return np.maximum(0, feature_map)


def relu_backward(feature_map, upstream_gradient):
    feature_map = np.array(feature_map, dtype=float)
    upstream_gradient = np.array(upstream_gradient, dtype=float)

    if feature_map.shape != upstream_gradient.shape:
        raise ValueError(
            "Feature map and upstream gradient must have the same shape"
        )

    # ReLU derivative:
    # positive input  -> 1
    # zero/negative   -> 0
    relu_derivative = np.where(feature_map > 0, 1.0, 0.0)

    input_gradient = upstream_gradient * relu_derivative

    return input_gradient