from network import NeuralNetwork


print("===== NEURAL NETWORK TEST =====")


# -------------------------
# INPUT
# -------------------------

input_size = int(
    input("How many inputs? ")
)

inputs = []

for i in range(input_size):

    value = float(
        input(f"Enter input x{i + 1}: ")
    )

    inputs.append(value)


# -------------------------
# NETWORK
# -------------------------

network = NeuralNetwork()


# First hidden layer
hidden_neurons = int(
    input("\nHow many neurons in hidden layer? ")
)

network.add_layer(
    input_size,
    hidden_neurons,
    "relu"
)


# Output layer
output_neurons = int(
    input("How many neurons in output layer? ")
)

network.add_layer(
    hidden_neurons,
    output_neurons,
    "sigmoid"
)


# -------------------------
# FORWARD PASS
# -------------------------

result = network.forward(inputs)


# -------------------------
# DISPLAY
# -------------------------

print("\n===== NETWORK RESULT =====")

for layer in result["layers"]:

    print(
        f"\n----- Layer {layer['layer']} -----"
    )

    print(
        "Inputs:",
        layer["inputs"]
    )

    print(
        "Weighted sums:",
        layer["weighted_sums"]
    )

    print(
        "Activation:",
        layer["activation"]
    )

    print(
        "Outputs:",
        layer["outputs"]
    )


print("\n===== FINAL OUTPUT =====")

print(
    result["final_output"]
)