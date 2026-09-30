import numpy as np

from .cell import RNNCell
from .output import RNNOutputLayer


class RNNModel:

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # RNN hidden-state layer
        self.cell = RNNCell(
            input_size=input_size,
            hidden_size=hidden_size
        )

        # Output layer
        self.output_layer = RNNOutputLayer(
            hidden_size=hidden_size
        )

    def forward(self, sequence, initial_hidden):
        """
        Run the sequence through the RNN.
        """

        from .forward import forward_sequence

        return forward_sequence(
            cell=self.cell,
            sequence=sequence,
            initial_hidden=initial_hidden
        )

    def predict(self, sequence, initial_hidden):
        """
        Run the RNN and produce a probability.
        """

        forward_result = self.forward(
            sequence=sequence,
            initial_hidden=initial_hidden
        )

        final_hidden = np.array(
            forward_result["final_hidden"],
            dtype=float
        )

        output_result = self.output_layer.forward(
            final_hidden
        )

        return {
            "final_hidden": final_hidden,
            "logit": output_result["logit"],
            "probability": output_result["probability"],
            "forward_history": forward_result["history"]
        }