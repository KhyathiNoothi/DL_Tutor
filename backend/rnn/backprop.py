import numpy as np

from .activation import tanh


def bptt(
    cell,
    output_layer,
    sequence,
    forward_history,
    target,
    probability
):
    """
    Backpropagation Through Time (BPTT).

    Calculates gradients for:

        Wxh
        Whh
        bh
        Why
        by
    """

    sequence = np.array(
        sequence,
        dtype=float
    )

    target = float(target)
    probability = float(probability)

    # --------------------------------------------------
    # Sigmoid + Binary Cross Entropy
    #
    # dL/dlogit = prediction - target
    # --------------------------------------------------

    dL_dlogit = probability - target

    # --------------------------------------------------
    # Output layer gradients
    # --------------------------------------------------

    final_hidden = np.array(
        forward_history[-1]["hidden"],
        dtype=float
    )

    dL_dWhy = (
        final_hidden * dL_dlogit
    )

    dL_dby = dL_dlogit

    # Gradient flowing back into final hidden state
    dh_next = (
        output_layer.weights * dL_dlogit
    )

    # --------------------------------------------------
    # Initialize RNN gradients
    # --------------------------------------------------

    dL_dWxh = np.zeros_like(
        cell.Wxh
    )

    dL_dWhh = np.zeros_like(
        cell.Whh
    )

    dL_dbh = np.zeros_like(
        cell.bh
    )

    # --------------------------------------------------
    # Backpropagate through time
    # --------------------------------------------------

    for time_step in reversed(
        range(len(sequence))
    ):

        step = forward_history[time_step]

        x = np.array(
            step["input"],
            dtype=float
        )

        previous_hidden = np.array(
            step["previous_hidden"],
            dtype=float
        )

        z = np.array(
            step["weighted_sum"],
            dtype=float
        )

        # tanh derivative
        tanh_derivative = (
            1 - np.tanh(z) ** 2
        )

        # Gradient through tanh
        dL_dz = (
            dh_next * tanh_derivative
        )

        # --------------------------------------------------
        # Gradients for input → hidden weights
        # --------------------------------------------------

        dL_dWxh += np.outer(
            x,
            dL_dz
        )

        # --------------------------------------------------
        # Gradients for hidden → hidden weights
        # --------------------------------------------------

        dL_dWhh += np.outer(
            previous_hidden,
            dL_dz
        )

        # --------------------------------------------------
        # Gradient for hidden bias
        # --------------------------------------------------

        dL_dbh += dL_dz

        # --------------------------------------------------
        # Send gradient to previous hidden state
        # --------------------------------------------------

        dh_next = np.dot(
            cell.Whh,
            dL_dz
        )

    return {
        "dL_dWxh": dL_dWxh.tolist(),
        "dL_dWhh": dL_dWhh.tolist(),
        "dL_dbh": dL_dbh.tolist(),
        "dL_dWhy": dL_dWhy.tolist(),
        "dL_dby": float(dL_dby),
        "dL_dlogit": float(dL_dlogit)
    }