import numpy as np

from prediction import predict_binary, predict_multiclass


print("======================================")
print("          CNN PREDICTION TEST")
print("======================================")


print("\nChoose prediction type:")
print("1. Binary Classification")
print("2. Multi-class Classification")


choice = int(input("\nEnter choice: "))


# --------------------------------
# BINARY CLASSIFICATION
# --------------------------------

if choice == 1:

    logit = float(input("\nEnter output logit: "))

    result = predict_binary(logit)

    print("\n======================================")
    print("        BINARY PREDICTION")
    print("======================================")

    print("Probability:", result["probability"])
    print("Predicted class:", result["predicted_class"])


# --------------------------------
# MULTI-CLASS CLASSIFICATION
# --------------------------------

elif choice == 2:

    num_classes = int(input("\nNumber of classes: "))

    logits = []

    print("\nEnter logits:")

    for i in range(num_classes):

        value = float(
            input(f"Logit for class {i}: ")
        )

        logits.append(value)

    result = predict_multiclass(logits)

    print("\n======================================")
    print("        MULTI-CLASS PREDICTION")
    print("======================================")

    print("Probabilities:")

    for i, probability in enumerate(
        result["probabilities"]
    ):
        print(
            f"Class {i}: {probability:.4f}"
        )

    print(
        "\nPredicted class:",
        result["predicted_class"]
    )

else:

    print("\nInvalid choice.")