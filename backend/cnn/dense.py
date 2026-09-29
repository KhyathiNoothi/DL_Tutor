import numpy as np


class DenseLayer:
    """
    Fully connected layer used after CNN flattening.
    """

    def __init__(self, input_size, output_size, activation="linear"):

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Initialize weights with small random values
        self.weights = np.random.randn(
            input_size,
            output_size
        ) * 0.01

        # One bias for each output neuron
        self.biases = np.zeros(output_size)


    def forward(self, inputs):

        inputs = np.array(inputs, dtype=float)

        if inputs.size != self.input_size:
            raise ValueError(
                "Number of inputs must match input_size"
            )

        # Weighted sum
        z = np.dot(inputs, self.weights) + self.biases

        # Activation
        if self.activation == "relu":
            output = np.maximum(0, z)

        elif self.activation == "sigmoid":
            output = 1 / (1 + np.exp(-z))

        elif self.activation == "linear":
            output = z

        else:
            raise ValueError(
                "Unsupported activation"
            )

        return {
            "input": inputs,
            "weighted_sum": z,
            "output": output
        }