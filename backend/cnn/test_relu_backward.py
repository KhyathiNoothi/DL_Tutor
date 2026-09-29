import numpy as np

from activation import relu_backward


print("======================================")
print("          RELU BACKPROP TEST")
print("======================================")


# Feature map before ReLU
feature_map = np.array([
    [-3, 2],
    [0, 5]
])


# Gradient coming from the next layer
upstream_gradient = np.array([
    [0.4, 0.7],
    [0.2, 0.9]
])


print("\n======================================")
print("       FEATURE MAP BEFORE RELU")
print("======================================")

print(feature_map)


print("\n======================================")
print("        UPSTREAM GRADIENT")
print("======================================")

print(upstream_gradient)


# ReLU backward
input_gradient = relu_backward(
    feature_map,
    upstream_gradient
)


print("\n======================================")
print("       GRADIENT AFTER RELU")
print("======================================")

print(input_gradient)