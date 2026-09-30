import numpy as np

from .activation import tanh


class RNNCell:

    def __init__(
        self,
        input_size,
        hidden_size
    ):

        self.input_size = input_size
        self.hidden_size = hidden_size

        limit_xh = np.sqrt(
            6 / (input_size + hidden_size)
        )

        limit_hh = np.sqrt(
            6 / (hidden_size + hidden_size)
        )

        self.Wxh = np.random.uniform(
            -limit_xh,
            limit_xh,
            (input_size, hidden_size)
        )

        self.Whh = np.random.uniform(
            -limit_hh,
            limit_hh,
            (hidden_size, hidden_size)
        )

        # Hidden bias
        self.bh = np.zeros(hidden_size)


    def forward(
        self,
        x,
        previous_hidden
    ):

        x = np.array(
            x,
            dtype=float
        )

        previous_hidden = np.array(
            previous_hidden,
            dtype=float
        )

        if len(x) != self.input_size:

            raise ValueError(
                "Input size does not match input_size"
            )

        if len(previous_hidden) != self.hidden_size:

            raise ValueError(
                "Hidden state size does not match hidden_size"
            )

        # Input contribution
        input_contribution = np.dot(
            x,
            self.Wxh
        )

        # Previous hidden-state contribution
        hidden_contribution = np.dot(
            previous_hidden,
            self.Whh
        )

        # Weighted sum
        z = (
            input_contribution
            + hidden_contribution
            + self.bh
        )

        # New hidden state
        hidden = tanh(z)

        return {
            "input": x,
            "previous_hidden": previous_hidden,
            "weighted_sum": z,
            "hidden": hidden
        }