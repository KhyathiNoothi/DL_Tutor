import numpy as np

from network import CNN
from training import train_cnn


print("======================================")
print("         CNN TRAINING TEST")
print("======================================")


# ==================================================
# 1. INPUT IMAGE
# ==================================================

height = int(input("\nEnter image height: "))
width = int(input("Enter image width: "))

image = []

print("\nEnter image values:")

for i in range(height):

    row = []

    for j in range(width):

        value = float(
            input(f"Pixel [{i}][{j}]: ")
        )

        row.append(value)

    image.append(row)


image = np.array(image, dtype=float)


# ==================================================
# 2. NUMBER OF FILTERS
# ==================================================

num_filters = int(
    input("\nEnter number of filters: ")
)


kernels = []


# ==================================================
# 3. ENTER FILTERS
# ==================================================

for filter_index in range(num_filters):

    print(
        f"\n========== FILTER {filter_index + 1} =========="
    )

    filter_height = int(
        input("Enter filter height: ")
    )

    filter_width = int(
        input("Enter filter width: ")
    )

    kernel = []

    print("\nEnter filter values:")

    for i in range(filter_height):

        row = []

        for j in range(filter_width):

            value = float(
                input(
                    f"Filter [{i}][{j}]: "
                )
            )

            row.append(value)

        kernel.append(row)

    kernels.append(
        np.array(kernel, dtype=float)
    )


# ==================================================
# 4. CONVOLUTION SETTINGS
# ==================================================

stride = int(
    input("\nEnter convolution stride: ")
)

padding = int(
    input("Enter convolution padding: ")
)


# ==================================================
# 5. POOLING SETTINGS
# ==================================================

pool_size = int(
    input("\nEnter pooling size: ")
)

pool_stride = int(
    input("Enter pooling stride: ")
)


# ==================================================
# 6. TARGET
# ==================================================

target = float(
    input("\nEnter target (0 or 1): ")
)


# ==================================================
# 7. LEARNING RATE
# ==================================================

learning_rate = float(
    input("Enter learning rate: ")
)


# ==================================================
# 8. EPOCHS
# ==================================================

epochs = int(
    input("Enter number of epochs: ")
)


# ==================================================
# 9. CREATE CNN
# ==================================================

cnn = CNN(
    kernels=kernels,
    pool_size=pool_size,
    pool_stride=pool_stride,
    dense_output_size=1,
    dense_activation="linear"
)


# ==================================================
# 10. INITIAL FILTERS
# ==================================================

print("\n======================================")
print("          INITIAL FILTERS")
print("======================================")

for i, kernel in enumerate(cnn.kernels):

    print(f"\nFilter {i + 1}:")
    print(kernel)


# ==================================================
# 11. TRAIN CNN
# ==================================================

print("\n======================================")
print("          STARTING TRAINING")
print("======================================\n")


result = train_cnn(
    cnn=cnn,
    image=image,
    target=target,
    learning_rate=learning_rate,
    epochs=epochs,
    stride=stride,
    padding=padding
)


# ==================================================
# 12. FINAL RESULT
# ==================================================

print("\n======================================")
print("          TRAINING COMPLETE")
print("======================================")


print(
    "\nFinal probability:",
    result["final_probability"]
)

print(
    "Final loss:",
    result["final_loss"]
)


# ==================================================
# 13. FINAL FILTERS
# ==================================================

print("\n======================================")
print("           FINAL FILTERS")
print("======================================")

for i, kernel in enumerate(cnn.kernels):

    print(f"\nFilter {i + 1}:")
    print(kernel)
# ==================================================
# 14. DISPLAY FIRST EPOCH DETAILS
# ==================================================

first_epoch = result["history"][0]

print("\n======================================")
print("        FIRST EPOCH DETAILS")
print("======================================")


print("\n--- FEATURE MAPS ---")

for i, feature_map in enumerate(
    first_epoch["feature_maps"]
):
    print(f"\nFilter {i + 1} Feature Map:")
    print(np.array(feature_map))


print("\n--- AFTER RELU ---")

for i, activated_map in enumerate(
    first_epoch["activated_maps"]
):
    print(f"\nFilter {i + 1} ReLU Output:")
    print(np.array(activated_map))


print("\n--- AFTER POOLING ---")

for i, pooled_map in enumerate(
    first_epoch["pooled_maps"]
):
    print(f"\nFilter {i + 1} Pooled Map:")
    print(np.array(pooled_map))


print("\n--- FLATTENED ---")

print(
    np.array(first_epoch["flattened"])
)


print("\n--- PREDICTION ---")

print(
    "Logit:",
    first_epoch["logit"]
)

print(
    "Probability:",
    first_epoch["probability"]
)


print("\n--- LOSS ---")

print(
    "Target:",
    first_epoch["target"]
)

print(
    "Loss:",
    first_epoch["loss"]
)


print("\n--- FILTER GRADIENTS ---")

for i, gradient in enumerate(
    first_epoch["kernel_gradients"]
):
    print(f"\nFilter {i + 1} Gradient:")
    print(np.array(gradient))


print("\n--- OLD FILTERS ---")

for i, kernel in enumerate(
    first_epoch["old_kernels"]
):
    print(f"\nFilter {i + 1}:")
    print(np.array(kernel))


print("\n--- UPDATED FILTERS ---")

for i, kernel in enumerate(
    first_epoch["new_kernels"]
):
    print(f"\nFilter {i + 1}:")
    print(np.array(kernel))