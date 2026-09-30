import numpy as np


def update_parameters(
    Wxh,
    Whh,
    bh,
    Why,
    by,
    dL_dWxh,
    dL_dWhh,
    dL_dbh,
    dL_dWhy,
    dL_dby,
    learning_rate
):
    """
    Update all RNN parameters using
    Gradient Descent.
    """

    if learning_rate <= 0:
        raise ValueError(
            "Learning rate must be greater than 0"
        )

    Wxh = np.array(Wxh, dtype=float)
    Whh = np.array(Whh, dtype=float)
    bh = np.array(bh, dtype=float)
    Why = np.array(Why, dtype=float)
    by = float(by)

    dL_dWxh = np.array(dL_dWxh, dtype=float)
    dL_dWhh = np.array(dL_dWhh, dtype=float)
    dL_dbh = np.array(dL_dbh, dtype=float)
    dL_dWhy = np.array(dL_dWhy, dtype=float)
    dL_dby = float(dL_dby)

    # Gradient Descent
    Wxh = Wxh - learning_rate * dL_dWxh

    Whh = Whh - learning_rate * dL_dWhh

    bh = bh - learning_rate * dL_dbh

    Why = Why - learning_rate * dL_dWhy

    by = by - learning_rate * dL_dby

    return {
        "Wxh": Wxh,
        "Whh": Whh,
        "bh": bh,
        "Why": Why,
        "by": by
    }