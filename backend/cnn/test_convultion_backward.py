import numpy as np

from convolution import convolution2d, convolution_backward


print("======================================")
print("      CONVOLUTION BACKPROP TEST")
print("======================================")


# Input image
image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])


# Filter
kernel = np.array([
    [1, 0],
    [0, 1]
])


# Forward convolution
feature_map = convolution2d(
    image,
    kernel,
    stride=1,
    padding=0
)


print("\n======================================")
print("             INPUT IMAGE")
print("======================================")

print(image)


print("\n======================================")
print("               KERNEL")
print("======================================")

print(kernel)


print("\n======================================")
print("           FEATURE MAP")
print("======================================")

print(feature_map)


# Gradient coming from the next layer
upstream_gradient = np.ones_like(feature_map)


print("\n======================================")
print("        UPSTREAM GRADIENT")
print("======================================")

print(upstream_gradient)


# Backward pass
result = convolution_backward(
    image,
    kernel,
    upstream_gradient,
    stride=1,
    padding=0
)


print("\n======================================")
print("         KERNEL GRADIENT")
print("======================================")

print(result["kernel_gradient"])


print("\n======================================")
print("          INPUT GRADIENT")
print("======================================")

print(result["input_gradient"])