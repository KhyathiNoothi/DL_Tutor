import numpy as np
from pooling import max_pooling2d


print("======================================")
print("          CNN MAX POOLING TEST")
print("======================================")


height = int(input("\nFeature map height: "))
width = int(input("Feature map width: "))


feature_map = []

print("\nEnter feature map values:")

for i in range(height):

    row = []

    for j in range(width):

        value = float(input(f"Value [{i}][{j}]: "))

        row.append(value)

    feature_map.append(row)


feature_map = np.array(feature_map)


pool_size = int(input("\nPool size: "))
stride = int(input("Stride: "))


print("\n======================================")
print("          INPUT FEATURE MAP")
print("======================================")

print(feature_map)


pooled_map = max_pooling2d(
    feature_map,
    pool_size,
    stride
)


print("\n======================================")
print("          AFTER MAX POOLING")
print("======================================")

print(pooled_map)