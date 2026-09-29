import numpy as np

from optimizer import update_parameters, update_kernel


print("======================================")
print("          CNN OPTIMIZER TEST")
print("======================================")


# ======================================
# TEST 1: DENSE PARAMETERS
# ======================================

weights = np.array([0.5, 0.2])

biases = np.array([0.0])

weight_gradients = np.array([1.0, 1.5])

bias_gradient = 0.5

learning_rate = 0.1


dense_result = update_parameters(
    weights,
    biases,
    weight_gradients,
    bias_gradient,
    learning_rate
)


print("\n======================================")
print("         DENSE UPDATE")
print("======================================")

print("Old weights:")
print(weights)

print("New weights:")
print(dense_result["weights"])

print("Old biases:")
print(biases)

print("New biases:")
print(dense_result["biases"])


# ======================================
# TEST 2: CNN KERNEL
# ======================================

kernel = np.array([
    [1.0, 0.5],
    [0.2, -0.3]
])


kernel_gradient = np.array([
    [0.4, 0.2],
    [0.1, -0.5]
])


learning_rate = 0.1


new_kernel = update_kernel(
    kernel,
    kernel_gradient,
    learning_rate
)


print("\n======================================")
print("          KERNEL UPDATE")
print("======================================")

print("Old kernel:")
print(kernel)

print("Kernel gradient:")
print(kernel_gradient)

print("Learning rate:")
print(learning_rate)

print("New kernel:")
print(new_kernel)