
import numpy as np

from backprop import flatten_backward


print("======================================")
print("       FLATTEN BACKPROP TEST")
print("======================================")


gradient = [0.2, 0.5, -0.1, 0.3]

original_shape = (2, 2)


result = flatten_backward(
    gradient,
    original_shape
)


print("\n======================================")
print("       FLATTENED GRADIENT")
print("======================================")

print(gradient)


print("\n======================================")
print("      GRADIENT AFTER UNFLATTEN")
print("======================================")

print(result)