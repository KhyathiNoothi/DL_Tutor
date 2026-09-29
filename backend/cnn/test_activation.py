import numpy as np

from activation import relu


print("======================================")
print("          CNN RELU TEST")
print("======================================")


# --------------------------------------
# 1. FEATURE MAP DIMENSIONS
# --------------------------------------

height = int(
    input("\nFeature map height: ")
)

width = int(
    input("Feature map width: ")
)


# --------------------------------------
# 2. ENTER FEATURE MAP
# --------------------------------------

feature_map = []

print("\nEnter feature map values:")

for i in range(height):

    row = []

    for j in range(width):

        value = float(
            input(f"Value [{i}][{j}]: ")
        )

        row.append(value)

    feature_map.append(row)


feature_map = np.array(
    feature_map
)


# --------------------------------------
# 3. DISPLAY INPUT
# --------------------------------------

print("\n======================================")
print("          INPUT FEATURE MAP")
print("======================================")

print(feature_map)


# --------------------------------------
# 4. APPLY RELU
# --------------------------------------

activated_map = relu(
    feature_map
)


# --------------------------------------
# 5. DISPLAY OUTPUT
# --------------------------------------

print("\n======================================")
print("          AFTER RELU")
print("======================================")

print(activated_map)