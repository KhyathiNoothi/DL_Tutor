import numpy as np

from .cell import RNNCell


def forward_sequence(
    cell,
    sequence,
    initial_hidden
):
    """
    Run an entire sequence through an RNN.

    sequence:
        A list/array containing inputs
        for each time step.

    initial_hidden:
        Hidden state before the first time step.
    """

    sequence = np.array(
        sequence,
        dtype=float
    )

    initial_hidden = np.array(
        initial_hidden,
        dtype=float
    )

    if sequence.ndim != 2:
        raise ValueError(
            "Sequence must be a 2D array"
        )

    if sequence.shape[1] != cell.input_size:
        raise ValueError(
            "Sequence input size does not "
            "match cell input_size"
        )

    if len(initial_hidden) != cell.hidden_size:
        raise ValueError(
            "Initial hidden state size does "
            "not match cell hidden_size"
        )

    hidden = initial_hidden

    history = []

    for time_step in range(len(sequence)):

        x = sequence[time_step]

        result = cell.forward(
            x,
            hidden
        )

        hidden = result["hidden"]

        history.append({
            "time_step": time_step + 1,
            "input": result["input"].tolist(),
            "previous_hidden": (
                result["previous_hidden"].tolist()
            ),
            "weighted_sum": (
                result["weighted_sum"].tolist()
            ),
            "hidden": (
                result["hidden"].tolist()
            )
        })

    return {
        "final_hidden": hidden.tolist(),
        "history": history
    }