import numpy as np


def bptt(
    cell,
    output_layer,
    sequence,
    forward_history,
    target,
    probability
):
    sequence = np.array(sequence, dtype=float)
    target = float(target)
    probability = float(probability)

    # -------------------------------------------------
    # 1. Gradient of BCE + Sigmoid
    # -------------------------------------------------

    dL_dlogit = probability - target

    # -------------------------------------------------
    # 2. Output layer gradients
    # -------------------------------------------------

    final_hidden = np.array(
        forward_history[-1]["hidden"],
        dtype=float
    )

    dL_dWhy = final_hidden * dL_dlogit
    dL_dby = dL_dlogit

    # Gradient flowing from output layer
    # into the final hidden state
    dh_next = output_layer.weights * dL_dlogit

    # -------------------------------------------------
    # 3. Initialize RNN gradients
    # -------------------------------------------------

    dL_dWxh = np.zeros_like(cell.Wxh)
    dL_dWhh = np.zeros_like(cell.Whh)
    dL_dbh = np.zeros_like(cell.bh)

    # -------------------------------------------------
    # 4. NEW:
    # Gradient for every input vector
    # -------------------------------------------------

    dL_dinputs = np.zeros_like(sequence)

    # -------------------------------------------------
    # 5. Backpropagation Through Time
    # -------------------------------------------------

    for time_step in reversed(range(len(sequence))):

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

        # -------------------------------------------------
        # Gradient for Wxh
        # -------------------------------------------------

        dL_dWxh += np.outer(
            x,
            dL_dz
        )

        # -------------------------------------------------
        # Gradient for Whh
        # -------------------------------------------------

        dL_dWhh += np.outer(
            previous_hidden,
            dL_dz
        )

        # -------------------------------------------------
        # Gradient for bias
        # -------------------------------------------------

        dL_dbh += dL_dz

        # -------------------------------------------------
        # NEW:
        # Gradient with respect to current input x
        #
        # z = x @ Wxh + h_previous @ Whh + bh
        #
        # Therefore:
        #
        # dL/dx = dL/dz @ Wxh.T
        # -------------------------------------------------

        dL_dx = np.dot(
            dL_dz,
            cell.Wxh.T
        )

        dL_dinputs[time_step] = dL_dx

        # -------------------------------------------------
        # Pass gradient backward through time
        # -------------------------------------------------

        dh_next = np.dot(
            cell.Whh,
            dL_dz
        )

    # -------------------------------------------------
    # 6. Return all gradients
    # -------------------------------------------------

    return {
        "dL_dWxh": dL_dWxh.tolist(),
        "dL_dWhh": dL_dWhh.tolist(),
        "dL_dbh": dL_dbh.tolist(),

        "dL_dWhy": dL_dWhy.tolist(),
        "dL_dby": float(dL_dby),

        "dL_dlogit": float(dL_dlogit),

        # NEW
        "dL_dinputs": dL_dinputs.tolist()
    }