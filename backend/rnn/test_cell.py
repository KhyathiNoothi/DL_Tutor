import numpy as np

from .cell import RNNCell


cell = RNNCell(
    input_size=2,
    hidden_size=3
)


x = np.array([1, 2])

previous_hidden = np.array([
    0,
    0,
    0
])


result = cell.forward(
    x,
    previous_hidden
)


print("Input:")
print(result["input"])

print("\nPrevious Hidden State:")
print(result["previous_hidden"])

print("\nWeighted Sum:")
print(result["weighted_sum"])

print("\nNew Hidden State:")
print(result["hidden"])