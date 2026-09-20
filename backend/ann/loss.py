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