import numpy as np

from .cell import RNNCell
from .training import train_rnn


# =========================================
# USER INPUT
# =========================================

input_size = int(
    input("Enter input size: ")
)

hidden_size = int(
    input("Enter hidden size: ")
)

sequence_length = int(
    input("Enter sequence length: ")
)


# =========================================
# ENTER SEQUENCE
# =========================================

sequence = []

print("\nEnter sequence values:")

for time_step in range(sequence_length):

    print(
        f"\nTime Step {time_step + 1}"
    )

    values = []

    for i in range(input_size):

        value = float(
            input(f"Enter x[{i}]: ")
        )

        values.append(value)

    sequence.append(values)


# =========================================
# INITIAL HIDDEN STATE
# =========================================

print("\nEnter initial hidden state:")

initial_hidden = []

for i in range(hidden_size):

    value = float(
        input(f"Enter h0[{i}]: ")
    )

    initial_hidden.append(value)


# =========================================
# TARGET
# =========================================

target = float(
    input("\nEnter target (0 or 1): ")
)


# =========================================
# LEARNING RATE
# =========================================

learning_rate = float(
    input("Enter learning rate: ")
)


# =========================================
# EPOCHS
# =========================================

epochs = int(
    input("Enter number of epochs: ")
)


# =========================================
# CREATE RNN
# =========================================

cell = RNNCell(
    input_size=input_size,
    hidden_size=hidden_size
)


# =========================================
# TRAIN
# =========================================

result = train_rnn(
    cell=cell,
    sequence=sequence,
    initial_hidden=initial_hidden,
    target=target,
    learning_rate=learning_rate,
    epochs=epochs
)


# =========================================
# FINAL RESULT
# =========================================

print("\n==============================")
print("FINAL RNN RESULT")
print("==============================")

print("\nFinal Hidden State:")

print(
    result["final_hidden"]
)

print("\nFinal Probability:")

print(
    result["final_probability"]
)

print("\nFinal Loss:")

print(
    result["final_loss"]
)