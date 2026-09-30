import numpy as np

from .cell import RNNCell
from .forward import forward_sequence


cell = RNNCell(
    input_size=2,
    hidden_size=3
)


sequence = np.array([
    [1, 2],
    [2, 1],
    [3, 2]
])


initial_hidden = np.array([
    0,
    0,
    0
])


result = forward_sequence(
    cell=cell,
    sequence=sequence,
    initial_hidden=initial_hidden
)


print("Final Hidden State:")
print(result["final_hidden"])


print("\nSequence History:")

for step in result["history"]:

    print(
        f"\nTime Step: {step['time_step']}"
    )

    print(
        "Input:",
        step["input"]
    )

    print(
        "Previous Hidden:",
        step["previous_hidden"]
    )

    print(
        "Weighted Sum:",
        step["weighted_sum"]
    )

    print(
        "Hidden:",
        step["hidden"]
    )