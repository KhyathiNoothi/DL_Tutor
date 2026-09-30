import numpy as np

from .cell import RNNCell
from .forward import forward_sequence
from .output import RNNOutputLayer


# -----------------------------------------
# User input
# -----------------------------------------

input_size = int(
    input("Enter input size: ")
)

hidden_size = int(
    input("Enter hidden size: ")
)

sequence_length = int(
    input("Enter sequence length: ")
)


# -----------------------------------------
# Get sequence values
# -----------------------------------------

sequence = []

print("\nEnter values for each time step:")

for time_step in range(sequence_length):

    print(f"\nTime step {time_step + 1}")

    values = []

    for i in range(input_size):

        value = float(
            input(f"Enter x[{i}]: ")
        )

        values.append(value)

    sequence.append(values)


sequence = np.array(
    sequence,
    dtype=float
)


# -----------------------------------------
# Initial hidden state
# -----------------------------------------

print("\nEnter initial hidden state:")

initial_hidden = []

for i in range(hidden_size):

    value = float(
        input(f"Enter h0[{i}]: ")
    )

    initial_hidden.append(value)


initial_hidden = np.array(
    initial_hidden,
    dtype=float
)


# -----------------------------------------
# Create RNN cell
# -----------------------------------------

cell = RNNCell(
    input_size=input_size,
    hidden_size=hidden_size
)


# -----------------------------------------
# Forward through sequence
# -----------------------------------------

forward_result = forward_sequence(
    cell=cell,
    sequence=sequence,
    initial_hidden=initial_hidden
)


final_hidden = np.array(
    forward_result["final_hidden"],
    dtype=float
)


# -----------------------------------------
# Output layer
# -----------------------------------------

output_layer = RNNOutputLayer(
    hidden_size=hidden_size
)


output_result = output_layer.forward(
    final_hidden
)


# -----------------------------------------
# Display
# -----------------------------------------

print("\n==============================")
print("RNN RESULT")
print("==============================")

print("\nFinal Hidden State:")
print(final_hidden)

print("\nLogit:")
print(output_result["logit"])

print("\nProbability:")
print(output_result["probability"])

if output_result["probability"] >= 0.5:
    print("\nPredicted Class: 1")
else:
    print("\nPredicted Class: 0")