import numpy as np


class Optimizer:
    """
    Optimizer for updating weights and biases.

    Supported optimizers:
    1. Gradient Descent
    2. Momentum
    3. Adam
    """

    def __init__(self, name="gradient_descent"):
        self.name = name.lower()

        # Momentum state
        self.velocity_weights = {}
        self.velocity_biases = {}

        # Adam state
        self.m_weights = {}
        self.m_biases = {}

        self.v_weights = {}
        self.v_biases = {}

        # Adam time step
        self.t = 0

    def update_layer_parameters(
        self,
        weights,
        biases,
        gradients,
        learning_rate,
        layer_index
    ):
        """
        Update weights and biases of one layer.
        """

        weights = np.array(weights, dtype=float)
        biases = np.array(biases, dtype=float)

        d_weights = np.array(
            gradients["dL_dweights"],
            dtype=float
        )

        d_biases = np.array(
            gradients["dL_dbiases"],
            dtype=float
        )

        # ==================================================
        # 1. GRADIENT DESCENT
        # ==================================================

        if self.name == "gradient_descent":

            new_weights = (
                weights
                - learning_rate * d_weights
            )

            new_biases = (
                biases
                - learning_rate * d_biases
            )

        # ==================================================
        # 2. MOMENTUM
        # ==================================================

        elif self.name == "momentum":

            momentum = 0.9

            # Initialize velocity for this layer
            if layer_index not in self.velocity_weights:

                self.velocity_weights[layer_index] = np.zeros_like(
                    weights
                )

                self.velocity_biases[layer_index] = np.zeros_like(
                    biases
                )

            # Update velocity
            self.velocity_weights[layer_index] = (
                momentum * self.velocity_weights[layer_index]
                - learning_rate * d_weights
            )

            self.velocity_biases[layer_index] = (
                momentum * self.velocity_biases[layer_index]
                - learning_rate * d_biases
            )

            # Update parameters
            new_weights = (
                weights
                + self.velocity_weights[layer_index]
            )

            new_biases = (
                biases
                + self.velocity_biases[layer_index]
            )

        # ==================================================
        # 3. ADAM
        # ==================================================

        elif self.name == "adam":

            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8

            # Initialize Adam state
            if layer_index not in self.m_weights:

                self.m_weights[layer_index] = np.zeros_like(
                    weights
                )

                self.m_biases[layer_index] = np.zeros_like(
                    biases
                )

                self.v_weights[layer_index] = np.zeros_like(
                    weights
                )

                self.v_biases[layer_index] = np.zeros_like(
                    biases
                )

            # First moment
            self.m_weights[layer_index] = (
                beta1 * self.m_weights[layer_index]
                + (1 - beta1) * d_weights
            )

            self.m_biases[layer_index] = (
                beta1 * self.m_biases[layer_index]
                + (1 - beta1) * d_biases
            )

            # Second moment
            self.v_weights[layer_index] = (
                beta2 * self.v_weights[layer_index]
                + (1 - beta2) * (d_weights ** 2)
            )

            self.v_biases[layer_index] = (
                beta2 * self.v_biases[layer_index]
                + (1 - beta2) * (d_biases ** 2)
            )

            # Bias correction
            m_hat_weights = (
                self.m_weights[layer_index]
                / (1 - beta1 ** self.t)
            )

            m_hat_biases = (
                self.m_biases[layer_index]
                / (1 - beta1 ** self.t)
            )

            v_hat_weights = (
                self.v_weights[layer_index]
                / (1 - beta2 ** self.t)
            )

            v_hat_biases = (
                self.v_biases[layer_index]
                / (1 - beta2 ** self.t)
            )

            # Adam update
            new_weights = (
                weights
                - learning_rate
                * m_hat_weights
                / (np.sqrt(v_hat_weights) + epsilon)
            )

            new_biases = (
                biases
                - learning_rate
                * m_hat_biases
                / (np.sqrt(v_hat_biases) + epsilon)
            )

        else:

            raise ValueError(
                "Optimizer must be "
                "'gradient_descent', 'momentum', or 'adam'"
            )

        return {
            "weights": new_weights.tolist(),
            "biases": new_biases.tolist()
        }