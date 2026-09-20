import numpy as np

from layer import DenseLayer


print("===== DENSE LAYER TEST =====")


# -------------------------
# USER INPUT
# -------------------------

input_size = int(
    input("How many inputs? ")
)

neuron_count = int(
    input("How many neurons? ")
)


# Get input values
inputs = []

for i in range(input_size):

    value = float(
        input(f"Enter input x{i + 1}: ")
    )

    inputs.append(value)


# -------------------------
# ACTIVATION
# -------------------------

print("\nChoose activation:")
print("1. ReLU")
print("2. Sigmoid")
print("3. Tanh")

choice = input("Enter choice: ")


if choice == "1":
    activation = "relu"

elif choice == "2":
    activation = "sigmoid"

elif choice == "3":
    activation = "tanh"

else:
    print("Invalid activation!")
    exit()


# -------------------------
# CREATE LAYER
# -------------------------

# Make random values reproducible
np.random.seed(42)

layer = DenseLayer(
    input_size,
    neuron_count,
    activation
)


# -------------------------
# FORWARD PASS
# -------------------------

result = layer.forward(inputs)


# -------------------------
# DISPLAY
# -------------------------

print("\n===== LAYER RESULT =====")

print("Inputs:")
print(result["inputs"])

print("\nWeights:")
print(np.array(result["weights"]))

print("\nBiases:")
print(result["biases"])

print("\nWeighted sums:")
print(result["weighted_sums"])

print("\nActivation:")
print(result["activation"])

print("\nOutputs:")
print(result["outputs"])