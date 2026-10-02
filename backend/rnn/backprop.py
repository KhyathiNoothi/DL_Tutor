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

    dL_dlogit = probability - target

    final_hidden = np.array(
        forward_history[-1]["hidden"],
        dtype=float
    )

    dL_dWhy = final_hidden * dL_dlogit
    dL_dby = dL_dlogit

    dh_next = output_layer.weights * dL_dlogit

    dL_dWxh = np.zeros_like(cell.Wxh)
    dL_dWhh = np.zeros_like(cell.Whh)
    dL_dbh = np.zeros_like(cell.bh)

    dL_dinputs = np.zeros_like(sequence)

    backward_history = []

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

        tanh_derivative = (
            1 - np.tanh(z) ** 2
        )

        dL_dz = (
            dh_next * tanh_derivative
        )

        dL_dWxh_step = np.outer(
            x,
            dL_dz
        )

        dL_dWhh_step = np.outer(
            previous_hidden,
            dL_dz
        )

        dL_dbh_step = dL_dz

        dL_dx = np.dot(
            dL_dz,
            cell.Wxh.T
        )

        dL_dinputs[time_step] = dL_dx

        backward_history.append({
            "time_step": time_step + 1,
            "input": x.tolist(),
            "previous_hidden": previous_hidden.tolist(),
            "weighted_sum": z.tolist(),
            "tanh_derivative": tanh_derivative.tolist(),
            "dh_next": dh_next.tolist(),
            "dL_dz": dL_dz.tolist(),
            "dL_dx": dL_dx.tolist(),
            "dL_dWxh_step": dL_dWxh_step.tolist(),
            "dL_dWhh_step": dL_dWhh_step.tolist(),
            "dL_dbh_step": dL_dbh_step.tolist()
        })

        dL_dWxh += dL_dWxh_step
        dL_dWhh += dL_dWhh_step
        dL_dbh += dL_dbh_step

        dh_next = np.dot(
            cell.Whh,
            dL_dz
        )

    backward_history.reverse()

    return {
        "dL_dWxh": dL_dWxh.tolist(),
        "dL_dWhh": dL_dWhh.tolist(),
        "dL_dbh": dL_dbh.tolist(),
        "dL_dWhy": dL_dWhy.tolist(),
        "dL_dby": float(dL_dby),
        "dL_dlogit": float(dL_dlogit),
        "dL_dinputs": dL_dinputs.tolist(),
        "backward_history": backward_history
    }