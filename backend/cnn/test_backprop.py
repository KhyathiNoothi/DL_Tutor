from backprop import dense_gradients


print("======================================")
print("        CNN DENSE BACKPROP TEST")
print("======================================")


inputs = [2, 3]

dL_dz = float(
    input("\nEnter dL/dz: ")
)


result = dense_gradients(
    inputs,
    dL_dz
)


print("\n======================================")
print("             GRADIENTS")
print("======================================")


print(
    "Weight gradients:",
    result["dL_dweights"]
)

print(
    "Bias gradient:",
    result["dL_dbias"]
)