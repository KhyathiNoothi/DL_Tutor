import numpy as np

from .cell import RNNCell
from .forward import forward_sequence
from .output import RNNOutputLayer
from .loss import binary_cross_entropy
from .backprop import bptt


# --------------------------------------------------
# Create RNN
# --------------------------------------------------

cell = RNNCell(
    input_size=2,
    hidden_size=3
)


# --------------------------------------------------
# Sequence
# --------------------------------------------------

sequence = np.array([
    [1, 2],
    [2, 1],
    [3, 2]
])


# --------------------------------------------------
# Initial hidden state
# --------------------------------------------------

initial_hidden = np.array([
    0,
    0,
    0
])


# --------------------------------------------------
# Target
# --------------------------------------------------

target = 0


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

forward_result = forward_sequence(
    cell=cell,
    sequence=sequence,
    initial_hidden=initial_hidden
)


final_hidden = np.array(
    forward_result["final_hidden"],
    dtype=float
)


# --------------------------------------------------
# Output layer
# --------------------------------------------------

output_layer = RNNOutputLayer(
    hidden_size=3
)


output_result = output_layer.forward(
    final_hidden
)


probability = output_result["probability"]


# --------------------------------------------------
# Loss
# --------------------------------------------------

loss = binary_cross_entropy(
    [target],
    [probability]
)


# --------------------------------------------------
# BPTT
# --------------------------------------------------

gradients = bptt(
    cell=cell,
    output_layer=output_layer,
    sequence=sequence,
    forward_history=forward_result["history"],
    target=target,
    probability=probability
)


# --------------------------------------------------
# Display
# --------------------------------------------------

print("Final Hidden State:")
print(final_hidden)


print("\nProbability:")
print(probability)


print("\nLoss:")
print(loss)


print("\nBPTT GRADIENTS")


print("\ndL/dWxh:")
print(
    np.array(
        gradients["dL_dWxh"]
    )
)


print("\ndL/dWhh:")
print(
    np.array(
        gradients["dL_dWhh"]
    )
)


print("\ndL/dbh:")
print(
    np.array(
        gradients["dL_dbh"]
    )
)


print("\ndL/dWhy:")
print(
    np.array(
        gradients["dL_dWhy"]
    )
)


print("\ndL/dby:")
print(
    gradients["dL_dby"]
)


print("\ndL/dlogit:")
print(
    gradients["dL_dlogit"]
)


# --------------------------------------------------
# NEW:
# Gradient with respect to each input
# --------------------------------------------------

print("\ndL/dinputs:")

for time_step, gradient in enumerate(
    gradients["dL_dinputs"]
):

    print(
        f"Time Step {time_step + 1}: "
        f"{gradient}"
    )