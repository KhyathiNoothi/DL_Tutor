import numpy as np

from loss import mean_squared_error
from optimizer import Optimizer


def train_network(
    network,
    inputs,
    actual,
    learning_rate,
    epochs,
    optimizer_name,
    backpropagation_method="standard"
):
    """
    Train a neural network using:

    1. Forward propagation
    2. Loss calculation
    3. Backpropagation
    4. Optimizer-based parameter updates

    Currently supported:
    - Backpropagation: standard
    - Optimizers: Gradient Descent, Momentum, Adam
    """

    # -----------------------------------
    # Validate backpropagation method
    # -----------------------------------

    if backpropagation_method != "standard":
        raise ValueError(
            "Currently only 'standard' backpropagation is supported"
        )

    history = []

    # -----------------------------------
    # Create optimizer
    # -----------------------------------

    optimizer = Optimizer(optimizer_name)

    # -----------------------------------
    # Training loop
    # -----------------------------------

    for epoch in range(epochs):

        # =================================
        # 1. FORWARD PROPAGATION
        # =================================

        forward_result = network.forward(inputs)

        prediction = np.array(
            forward_result["final_output"],
            dtype=float
        )

        actual_array = np.array(
            actual,
            dtype=float
        )

        # =================================
        # 2. LOSS CALCULATION
        # =================================

        loss = mean_squared_error(
            actual_array,
            prediction
        )

        # =================================
        # 3. LOSS GRADIENT
        # =================================

        output_gradient = 2 * (
            prediction - actual_array
        )

        # =================================
        # 4. BACKPROPAGATION
        # =================================

        gradients = network.backward(
            forward_result,
            output_gradient
        )

        # =================================
        # 5. UPDATE ADAM TIME STEP
        # =================================

        if optimizer.name == "adam":
            optimizer.t += 1

        layer_updates = []

        # =================================
        # 6. OPTIMIZER / WEIGHT UPDATE
        # =================================

        for layer_index, layer in enumerate(
            network.layers
        ):

            old_weights = layer.weights.copy()
            old_biases = layer.biases.copy()

            updated = optimizer.update_layer_parameters(
                layer.weights,
                layer.biases,
                gradients[layer_index],
                learning_rate,
                layer_index
            )

            layer.weights = np.array(
                updated["weights"],
                dtype=float
            )

            layer.biases = np.array(
                updated["biases"],
                dtype=float
            )

            layer_updates.append({
                "layer": layer_index + 1,

                "old_weights":
                    old_weights.tolist(),

                "old_biases":
                    old_biases.tolist(),

                "new_weights":
                    layer.weights.tolist(),

                "new_biases":
                    layer.biases.tolist(),

                "gradients":
                    gradients[layer_index]
            })

        # =================================
        # 7. SAVE TRAINING HISTORY
        # =================================

        history.append({
            "epoch": epoch + 1,

            "prediction":
                prediction.tolist(),

            "actual":
                actual_array.tolist(),

            "loss":
                float(loss),

            "forward":
                forward_result,

            "gradients":
                gradients,

            "updates":
                layer_updates
        })

    # -----------------------------------
    # FINAL RESULT
    # -----------------------------------

    return {
        "final_output":
            prediction.tolist(),

        "final_loss":
            float(loss),

        "history":
            history,

        "optimizer":
            optimizer.name,

        "backpropagation":
            backpropagation_method
    }