from forward import forward_pass


print("===== ANN FORWARD PASS =====")

# Number of inputs
n = int(input("How many inputs do you want? "))

# Get inputs
inputs = []

for i in range(n):
    value = float(input(f"Enter input x{i + 1}: "))
    inputs.append(value)


# Get weights
weights = []

for i in range(n):
    value = float(input(f"Enter weight w{i + 1}: "))
    weights.append(value)


# Get bias
bias = float(input("Enter bias: "))


# Choose activation
print("\nChoose activation function:")
print("1. ReLU")
print("2. Sigmoid")
print("3. Tanh")

choice = input("Enter your choice (1/2/3): ")

if choice == "1":
    activation = "relu"

elif choice == "2":
    activation = "sigmoid"

elif choice == "3":
    activation = "tanh"

else:
    print("Invalid choice!")
    exit()


# Perform forward pass
result = forward_pass(
    inputs,
    weights,
    bias,
    activation
)


# Display result
print("\n===== FORWARD PASS RESULT =====")

print("Inputs       :", result["inputs"])
print("Weights      :", result["weights"])
print("Bias         :", result["bias"])
print("Weighted Sum :", result["weighted_sum"])
print("Activation   :", result["activation"])
print("Output       :", result["output"])