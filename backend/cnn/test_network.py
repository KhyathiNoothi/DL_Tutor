import numpy as np

from network import CNN


print("======================================")
print("          COMPLETE CNN TEST")
print("======================================")


# --------------------------------
# IMAGE
# --------------------------------

height = int(input("\nImage height: "))
width = int(input("Image width: "))

image = []

print("\nEnter image values:")

for i in range(height):

    row = []

    for j in range(width):

        value = float(
            input(f"Image [{i}][{j}]: ")
        )

        row.append(value)

    image.append(row)

image = np.array(image)


# --------------------------------
# FILTER
# --------------------------------

num_filters = int(
    input("\nNumber of filters: ")
)

kernels = []


for f in range(num_filters):

    print(f"\n========== FILTER {f + 1} ==========")

    kernel_height = int(
        input("Filter height: ")
    )

    kernel_width = int(
        input("Filter width: ")
    )

    kernel = []

    print("\nEnter filter values:")

    for i in range(kernel_height):

        row = []

        for j in range(kernel_width):

            value = float(
                input(
                    f"Filter [{i}][{j}]: "
                )
            )

            row.append(value)

        kernel.append(row)

    kernels.append(
        np.array(kernel)
    )


# --------------------------------
# CNN SETTINGS
# --------------------------------

stride = int(input("\nConvolution stride: "))

padding = int(input("Convolution padding: "))

pool_size = int(input("Pool size: "))

pool_stride = int(input("Pool stride: "))


# --------------------------------
# CREATE CNN
# --------------------------------

cnn = CNN(
    kernels=kernels,
    pool_size=pool_size,
    pool_stride=pool_stride,
    dense_output_size=1,
    dense_activation="linear"
)


# --------------------------------
# FORWARD PASS
# --------------------------------

result = cnn.forward(
    image,
    stride,
    padding
)


# --------------------------------
# DISPLAY
# --------------------------------

print("\n======================================")
print("             INPUT IMAGE")
print("======================================")

print(image)


print("\n======================================")
print("          FEATURE MAPS")
print("======================================")

for i, feature_map in enumerate(
    result["feature_maps"]
):

    print(f"\nFeature Map {i + 1}:")
    print(feature_map)


print("\n======================================")
print("          AFTER RELU")
print("======================================")

for i, activated_map in enumerate(
    result["activated_maps"]
):

    print(f"\nActivated Map {i + 1}:")
    print(activated_map)


print("\n======================================")
print("          AFTER POOLING")
print("======================================")

for i, pooled_map in enumerate(
    result["pooled_maps"]
):

    print(f"\nPooled Map {i + 1}:")
    print(pooled_map)


print("\n======================================")
print("             FLATTENED")
print("======================================")

print(result["flattened"])


print("\n======================================")
print("           DENSE OUTPUT")
print("======================================")

print(result["dense"]["output"])
# --------------------------------
# PREDICTION
# --------------------------------

prediction = cnn.predict(
    result["dense"],
    task="binary"
)

print("\n======================================")
print("            PREDICTION")
print("======================================")

print("Probability:", prediction["probability"])
print("Predicted class:", prediction["predicted_class"])