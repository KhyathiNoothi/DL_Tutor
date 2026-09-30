import numpy as np


def binary_cross_entropy(actual, predicted):
    """
    Calculate Binary Cross-Entropy loss.

    actual:
        True target value (0 or 1).

    predicted:
        Predicted probability between 0 and 1.
    """

    actual = np.array(
        actual,
        dtype=float
    )

    predicted = np.array(
        predicted,
        dtype=float
    )

    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted must have "
            "the same shape"
        )

    # Prevent log(0)
    epsilon = 1e-7

    predicted = np.clip(
        predicted,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        actual * np.log(predicted)
        + (1 - actual) * np.log(1 - predicted)
    )

    return float(loss)