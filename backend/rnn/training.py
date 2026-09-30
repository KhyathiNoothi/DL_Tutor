import numpy as np

from .forward import forward_sequence
from .output import RNNOutputLayer
from .loss import binary_cross_entropy
from .backprop import bptt
from .optimizer import update_parameters


def train_rnn(
    cell,
    sequence,
    initial_hidden,
    target,
    learning_rate,
    epochs
):
    """
    Train an RNN using:

        Forward Pass
        ↓
        Prediction
        ↓
        Loss
        ↓
        BPTT
        ↓
        Gradients
        ↓
        Gradient Descent

    All values are supplied by the caller.
    """

    # -----------------------------------------
    # Convert user inputs to NumPy arrays
    # -----------------------------------------

    sequence = np.array(
        sequence,
        dtype=float
    )

    initial_hidden = np.array(
        initial_hidden,
        dtype=float
    )

    target = float(target)

    # -----------------------------------------
    # Validate inputs
    # -----------------------------------------

    if sequence.ndim != 2:
        raise ValueError(
            "Sequence must be a 2D array"
        )

    if sequence.shape[1] != cell.input_size:
        raise ValueError(
            "Sequence input size does not "
            "match the RNN input size"
        )

    if len(initial_hidden) != cell.hidden_size:
        raise ValueError(
            "Initial hidden state size does not "
            "match the RNN hidden size"
        )

    if target not in [0, 1]:
        raise ValueError(
            "Target must be 0 or 1"
        )

    if learning_rate <= 0:
        raise ValueError(
            "Learning rate must be greater than 0"
        )

    if epochs <= 0:
        raise ValueError(
            "Epochs must be greater than 0"
        )

    # -----------------------------------------
    # Create output layer
    # -----------------------------------------

    output_layer = RNNOutputLayer(
        hidden_size=cell.hidden_size
    )

    history = []

    # -----------------------------------------
    # Training loop
    # -----------------------------------------

    for epoch in range(epochs):

        # =====================================
        # 1. FORWARD PASS
        # =====================================

        forward_result = forward_sequence(
            cell=cell,
            sequence=sequence,
            initial_hidden=initial_hidden
        )

        final_hidden = np.array(
            forward_result["final_hidden"],
            dtype=float
        )

        # =====================================
        # 2. OUTPUT / PREDICTION
        # =====================================

        output_result = output_layer.forward(
            final_hidden
        )

        probability = output_result[
            "probability"
        ]

        logit = output_result[
            "logit"
        ]

        # =====================================
        # 3. LOSS
        # =====================================

        loss = binary_cross_entropy(
            [target],
            [probability]
        )

        # =====================================
        # 4. BPTT
        # =====================================

        gradients = bptt(
            cell=cell,
            output_layer=output_layer,
            sequence=sequence,
            forward_history=forward_result[
                "history"
            ],
            target=target,
            probability=probability
        )

        # =====================================
        # Save old parameters
        # =====================================

        old_Wxh = cell.Wxh.copy()
        old_Whh = cell.Whh.copy()
        old_bh = cell.bh.copy()

        old_Why = output_layer.weights.copy()
        old_by = output_layer.bias

        # =====================================
        # 5. UPDATE PARAMETERS
        # =====================================

        updated_parameters = update_parameters(
            Wxh=cell.Wxh,
            Whh=cell.Whh,
            bh=cell.bh,
            Why=output_layer.weights,
            by=output_layer.bias,

            dL_dWxh=gradients["dL_dWxh"],
            dL_dWhh=gradients["dL_dWhh"],
            dL_dbh=gradients["dL_dbh"],
            dL_dWhy=gradients["dL_dWhy"],
            dL_dby=gradients["dL_dby"],

            learning_rate=learning_rate
        )

        # =====================================
        # Put updated parameters back
        # =====================================

        cell.Wxh = updated_parameters["Wxh"]
        cell.Whh = updated_parameters["Whh"]
        cell.bh = updated_parameters["bh"]

        output_layer.weights = (
            updated_parameters["Why"]
        )

        output_layer.bias = (
            updated_parameters["by"]
        )

        # =====================================
        # Store complete epoch history
        # =====================================

        history.append({

            "epoch": epoch + 1,

            "sequence": sequence.tolist(),

            "initial_hidden": (
                initial_hidden.tolist()
            ),

            "hidden_states": [
                step["hidden"]
                for step in forward_result[
                    "history"
                ]
            ],

            "forward_history":
                forward_result["history"],

            "final_hidden":
                final_hidden.tolist(),

            "logit":
                float(logit),

            "probability":
                float(probability),

            "target":
                float(target),

            "loss":
                float(loss),

            "gradients": {
                "dL_dWxh":
                    gradients["dL_dWxh"],

                "dL_dWhh":
                    gradients["dL_dWhh"],

                "dL_dbh":
                    gradients["dL_dbh"],

                "dL_dWhy":
                    gradients["dL_dWhy"],

                "dL_dby":
                    gradients["dL_dby"],

                "dL_dlogit":
                    gradients["dL_dlogit"]
            },

            "old_parameters": {
                "Wxh":
                    old_Wxh.tolist(),

                "Whh":
                    old_Whh.tolist(),

                "bh":
                    old_bh.tolist(),

                "Why":
                    old_Why.tolist(),

                "by":
                    float(old_by)
            },

            "new_parameters": {
                "Wxh":
                    cell.Wxh.tolist(),

                "Whh":
                    cell.Whh.tolist(),

                "bh":
                    cell.bh.tolist(),

                "Why":
                    output_layer.weights.tolist(),

                "by":
                    float(output_layer.bias)
            }
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Probability: {probability:.6f} | "
            f"Loss: {loss:.6f}"
        )

    # -----------------------------------------
    # Return final result
    # -----------------------------------------

    return {

        "final_probability":
            float(probability),

        "final_loss":
            float(loss),

        "final_hidden":
            final_hidden.tolist(),

        "history":
            history
    }