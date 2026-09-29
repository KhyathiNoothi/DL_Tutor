import numpy as np
from dense import DenseLayer


print("======================================")
print("            CNN DENSE TEST")
print("======================================")


# -----------------------------
# INPUT
# -----------------------------

input_size = int(input("\nNumber of flattened inputs: "))

inputs = []

print("\nEnter flattened values:")

for i in range(input_size):

    value = float(input(f"Input [{i}]: "))

    inputs.append(value)


# -----------------------------
# DENSE LAYER
# -----------------------------

output_size = int(input("\nNumber of output neurons: "))

activation = input(
    "Activation (linear/relu/sigmoid): "
).strip().lower()


layer = DenseLayer(
    input_size=input_size,
    output_size=output_size,
    activation=activation
)


# -----------------------------
# DISPLAY WEIGHTS
# -----------------------------

print("\n======================================")
print("              WEIGHTS")
print("======================================")

print(layer.weights)


print("\n======================================")
print("              BIASES")
print("======================================")

print(layer.biases)


# -----------------------------
# FORWARD PASS
# -----------------------------

result = layer.forward(inputs)


print("\n======================================")
print("          WEIGHTED SUM")
print("======================================")

print(result["weighted_sum"])


print("\n======================================")
print("              OUTPUT")
print("======================================")

print(result["output"])