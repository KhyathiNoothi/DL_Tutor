from optimizer import update_parameters


print("======================================")
print("        CNN OPTIMIZER TEST")
print("======================================")


weights = [0.5, 0.2]
biases = [0.0]

weight_gradients = [1.0, 1.5]
bias_gradient = 0.5

learning_rate = 0.1


result = update_parameters(
    weights,
    biases,
    weight_gradients,
    bias_gradient,
    learning_rate
)


print("\n======================================")
print("          BEFORE UPDATE")
print("======================================")

print("Weights:", weights)
print("Biases:", biases)


print("\n======================================")
print("          AFTER UPDATE")
print("======================================")

print("Weights:", result["weights"])
print("Biases:", result["biases"])