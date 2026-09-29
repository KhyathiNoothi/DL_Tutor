import numpy as np

from convolution import convolution2d


print("======================================")
print("         CNN CONVOLUTION TEST")
print("======================================")


# --------------------------------------
# 1. INPUT IMAGE
# --------------------------------------

print("\nEnter Image Dimensions")

image_height = int(
    input("Image height: ")
)

image_width = int(
    input("Image width: ")
)

image = []

print("\nEnter image values:")

for i in range(image_height):

    row = []

    for j in range(image_width):

        value = float(
            input(f"Pixel [{i}][{j}]: ")
        )

        row.append(value)

    image.append(row)


image = np.array(image)


# --------------------------------------
# 2. KERNEL
# --------------------------------------

print("\nEnter Kernel Dimensions")

kernel_height = int(
    input("Kernel height: ")
)

kernel_width = int(
    input("Kernel width: ")
)

kernel = []

print("\nEnter kernel values:")

for i in range(kernel_height):

    row = []

    for j in range(kernel_width):

        value = float(
            input(f"Kernel [{i}][{j}]: ")
        )

        row.append(value)

    kernel.append(row)


kernel = np.array(kernel)


# --------------------------------------
# 3. STRIDE
# --------------------------------------

print("\nEnter Convolution Settings")

stride = int(
    input("Stride: ")
)


# --------------------------------------
# 4. PADDING
# --------------------------------------

padding = int(
    input("Padding: ")
)


# --------------------------------------
# 5. DISPLAY INPUTS
# --------------------------------------

print("\n======================================")
print("              INPUT")
print("======================================")

print("\nImage:")
print(image)

print("\nKernel:")
print(kernel)

print("\nStride:")
print(stride)

print("\nPadding:")
print(padding)


# --------------------------------------
# 6. CONVOLUTION
# --------------------------------------

feature_map = convolution2d(
    image,
    kernel,
    stride=stride,
    padding=padding
)


# --------------------------------------
# 7. OUTPUT
# --------------------------------------

print("\n======================================")
print("           FEATURE MAP")
print("======================================")

print(feature_map)