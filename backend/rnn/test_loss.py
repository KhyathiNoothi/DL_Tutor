from .loss import binary_cross_entropy


actual = [0]
predicted = [0.5]

loss = binary_cross_entropy(
    actual,
    predicted
)

print("Actual:")
print(actual)

print("\nPredicted:")
print(predicted)

print("\nBinary Cross-Entropy Loss:")
print(loss)