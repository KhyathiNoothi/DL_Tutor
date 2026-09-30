import numpy as np

from .activations import (
    relu,
    sigmoid,
    tanh,
    linear
)


class DenseLayer:
    """
    Fully connected (Dense) neural network layer.

    Each neuron receives all outputs
    from the previous layer.
    """

    def __init__(
        self,
        input_size,
        neuron_count,
        activation="relu"
    ):

        self.input_size = input_size
        self.neuron_count = neuron_count
        self.activation = activation

        # Weight matrix
        #
        # Shape:
        # input_size × neuron_count
        #
        self.weights = (
            np.random.randn(
                input_size,
                neuron_count
            ) * 0.1
        )

        # One bias for each neuron

        self.biases = np.zeros(
            neuron_count
        )

    def forward(self, inputs):

        # Convert inputs to NumPy array

        inputs = np.array(
            inputs,
            dtype=float
        )

        # --------------------------------
        # Check input size
        # --------------------------------

        if len(inputs) != self.input_size:

            raise ValueError(
                f"Expected {self.input_size} inputs, "
                f"but received {len(inputs)}"
            )

        # --------------------------------
        # Weighted sum
        # --------------------------------
        #
        # z = XW + b
        #

        z = (
            np.dot(
                inputs,
                self.weights
            )
            + self.biases
        )

        # --------------------------------
        # Activation function
        # --------------------------------

        if self.activation == "relu":

            output = relu(z)

        elif self.activation == "sigmoid":

            output = sigmoid(z)

        elif self.activation == "tanh":

            output = tanh(z)

        elif self.activation == "linear":

            output = linear(z)

        else:

            raise ValueError(
                "Activation must be "
                "'relu', 'sigmoid', "
                "'tanh', or 'linear'"
            )

        # --------------------------------
        # RETURN RESULT
        # --------------------------------

        return {
            "inputs": inputs.tolist(),

            "weights":
                self.weights.tolist(),

            "biases":
                self.biases.tolist(),

            "weighted_sums":
                z.tolist(),

            "activation":
                self.activation,

            "outputs":
                output.tolist()
        }