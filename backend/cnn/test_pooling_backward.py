import numpy as np

from pooling import max_pooling2d, max_pooling_backward


print("======================================")
print("       MAX POOLING BACKPROP TEST")
print("======================================")


# Original feature map
feature_map = np.array([
    [1, 5],
    [2, 3]
])


# Forward pooling
pooled_output, cache = max_pooling2d(
    feature_map,
    pool_size=2,
    stride=2,
    return_cache=True
)


print("\n======================================")
print("          ORIGINAL FEATURE MAP")
print("======================================")

print(feature_map)


print("\n======================================")
print("           POOLED OUTPUT")
print("======================================")

print(pooled_output)


# Suppose gradient coming from next layer
pooled_gradient = np.array([
    [0.8]
])


print("\n======================================")
print("        GRADIENT FROM NEXT LAYER")
print("======================================")

print(pooled_gradient)


# Backward pass
input_gradient = max_pooling_backward(
    pooled_gradient,
    cache
)


print("\n======================================")
print("       GRADIENT AFTER POOLING")
print("======================================")

print(input_gradient)