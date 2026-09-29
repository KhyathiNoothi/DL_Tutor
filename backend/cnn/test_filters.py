import numpy as np
from filters import apply_multiple_filters


print("======================================")
print("        CNN MULTIPLE FILTER TEST")
print("======================================")


# -----------------------------
# INPUT IMAGE
# -----------------------------

height = int(input("\nImage height: "))
width = int(input("Image width: "))

image = []

print("\nEnter image values:")

for i in range(height):

    row = []

    for j in range(width):

        value = float(input(f"Image [{i}][{j}]: "))

        row.append(value)

    image.append(row)

image = np.array(image)


# -----------------------------
# FILTERS
# -----------------------------

num_filters = int(input("\nNumber of filters: "))

kernels = []

for f in range(num_filters):

    print(f"\n========== FILTER {f + 1} ==========")

    kernel_height = int(input("Filter height: "))
    kernel_width = int(input("Filter width: "))

    kernel = []

    print("\nEnter filter values:")

    for i in range(kernel_height):

        row = []

        for j in range(kernel_width):

            value = float(
                input(f"Filter [{i}][{j}]: ")
            )

            row.append(value)

        kernel.append(row)

    kernels.append(np.array(kernel))


# -----------------------------
# STRIDE + PADDING
# -----------------------------

stride = int(input("\nStride: "))
padding = int(input("Padding: "))


# -----------------------------
# APPLY FILTERS
# -----------------------------

feature_maps = apply_multiple_filters(
    image,
    kernels,
    stride,
    padding
)


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

print("\n======================================")
print("             INPUT IMAGE")
print("======================================")

print(image)


for i, feature_map in enumerate(feature_maps):

    print("\n======================================")
    print(f"          FEATURE MAP {i + 1}")
    print("======================================")

    print(feature_map)