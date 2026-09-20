from activations import relu, sigmoid, tanh


value = float(input("Enter a value: "))

print("\nResults:")
print("Input   :", value)
print("ReLU    :", relu(value))
print("Sigmoid :", sigmoid(value))
print("Tanh    :", tanh(value))