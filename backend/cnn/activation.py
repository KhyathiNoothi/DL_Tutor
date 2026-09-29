import numpy as np


def relu(feature_map):
    """
    Apply ReLU activation element-by-element.

    ReLU(x) = max(0, x)
    """

    feature_map = np.array(
        feature_map,
        dtype=float
    )

    return np.maximum(
        0,
        feature_map
    )