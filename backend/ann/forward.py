from neuron import weighted_sum
from activations import relu, sigmoid, tanh


def forward_pass(inputs, weights, bias, activation):
    """
    Perform one complete forward pass through a neuron.
    Steps:
    1. Calculate weighted sum
    2. Add bias
    3. Apply activation function
    4. Return all intermediate values
    """

    # Step 1 + 2: weighted sum + bias
    z = weighted_sum(inputs, weights, bias)

    # Step 3: activation
    if activation == "relu":
        output = relu(z)

    elif activation == "sigmoid":
        output = sigmoid(z)

    elif activation == "tanh":
        output = tanh(z)

    else:
        raise ValueError(
            "Activation must be 'relu', 'sigmoid', or 'tanh'"
        )

    # Step 4: return everything
    return {
        "inputs": inputs,
        "weights": weights,
        "bias": bias,
        "weighted_sum": float(z),
        "activation": activation,
        "output": float(output)
    }