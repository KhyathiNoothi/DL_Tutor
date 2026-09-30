import numpy as np

from .activation import sigmoid


class RNNOutputLayer:

    def __init__(self, hidden_size):

        self.hidden_size = hidden_size

        # Hidden state → output weight
        self.weights = (
            np.random.randn(hidden_size) * 0.01
        )

        # Output bias
        self.bias = 0.0


    def forward(self, hidden):

        hidden = np.array(
            hidden,
            dtype=float
        )

        if len(hidden) != self.hidden_size:

            raise ValueError(
                "Hidden state size does not "
                "match hidden_size"
            )

        # Calculate output logit
        logit = (
            np.dot(hidden, self.weights)
            + self.bias
        )

        # Convert logit to probability
        probability = sigmoid(logit)

        return {
            "hidden": hidden,
            "logit": float(logit),
            "probability": float(probability)
        }