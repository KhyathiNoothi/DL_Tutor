import numpy as np

from layer import DenseLayer
from backprop import calculate_layer_gradients


class NeuralNetwork:
    """
    A simple feed-forward neural network
    made from multiple Dense layers.
    """

    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, neuron_count, activation="relu"):
        """
        Add a Dense layer to the network.
        """

        layer = DenseLayer(
            input_size,
            neuron_count,
            activation
        )

        self.layers.append(layer)

    def forward(self, inputs):
        """
        Pass the input through every layer.
        """

        current_input = inputs
        history = []

        for index, layer in enumerate(self.layers):

            result = layer.forward(current_input)

            history.append({
                "layer": index + 1,
                "inputs": result["inputs"],
                "weights": result["weights"],
                "biases": result["biases"],
                "weighted_sums": result["weighted_sums"],
                "activation": result["activation"],
                "outputs": result["outputs"]
            })

            # Output of current layer
            # becomes input to next layer
            current_input = result["outputs"]

        return {
            "final_output": current_input,
            "layers": history
        }

    def backward(self, forward_result, output_gradient):
        """
        Perform backpropagation through all layers.

        Backpropagation starts from the output layer
        and moves backwards toward the first layer.
        """

        gradients = []

        upstream_gradient = np.array(
            output_gradient,
            dtype=float
        )

        # Go through layers backwards
        for layer_index in range(
            len(self.layers) - 1,
            -1,
            -1
        ):

            layer = self.layers[layer_index]

            layer_history = forward_result["layers"][layer_index]

            inputs = np.array(
                layer_history["inputs"],
                dtype=float
            )

            weights = np.array(
                layer_history["weights"],
                dtype=float
            )

            biases = np.array(
                layer_history["biases"],
                dtype=float
            )

            z = np.array(
                layer_history["weighted_sums"],
                dtype=float
            )

            activation = layer_history["activation"]

            # Calculate gradients for this layer
            layer_gradient = calculate_layer_gradients(
                inputs,
                weights,
                biases,
                z,
                upstream_gradient,
                activation
            )

            gradients.insert(
                0,
                {
                    "layer": layer_index + 1,
                    "dL_dweights":
                        layer_gradient["dL_dweights"],
                    "dL_dbiases":
                        layer_gradient["dL_dbiases"],
                    "dL_dz":
                        layer_gradient["dL_dz"]
                }
            )

            # Calculate gradient that goes
            # to the previous layer
            dL_dz = np.array(
                layer_gradient["dL_dz"],
                dtype=float
            )

            upstream_gradient = np.dot(
                weights,
                dL_dz
            )

        return gradients