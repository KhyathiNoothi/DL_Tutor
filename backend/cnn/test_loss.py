from loss import binary_cross_entropy


print("======================================")
print("           CNN LOSS TEST")
print("======================================")


actual = float(
    input("\nActual class (0 or 1): ")
)

predicted = float(
    input("Predicted probability: ")
)


loss = binary_cross_entropy(
    [actual],
    [predicted]
)


print("\n======================================")
print("              RESULT")
print("======================================")

print("Actual:", actual)
print("Predicted probability:", predicted)
print("Loss:", loss)