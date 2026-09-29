import numpy as np


def mean_squared_error(actual, predicted):
    """
    Calculate Mean Squared Error.

    MSE = average((actual - predicted)^2)
    """

    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values must have the same shape")

    loss = np.mean((actual - predicted) ** 2)

    return float(loss)

def mean_squared_error(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted values must have the same shape"
        )

    loss = np.mean((actual - predicted) ** 2)

    return float(loss)


def binary_cross_entropy(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted values must have the same shape"
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