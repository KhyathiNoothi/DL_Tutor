import numpy as np
from flatten import flatten


print("======================================")
print("             CNN FLATTEN TEST")
print("======================================")


num_maps = int(input("\nNumber of feature maps: "))

feature_maps = []


for map_number in range(num_maps):

    print(f"\n========== FEATURE MAP {map_number + 1} ==========")

    height = int(input("Feature map height: "))
    width = int(input("Feature map width: "))

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

    feature_maps.append(np.array(feature_map))


print("\n======================================")
print("          INPUT FEATURE MAPS")
print("======================================")

for i, feature_map in enumerate(feature_maps):

    print(f"\nFeature Map {i + 1}:")
    print(feature_map)


flattened = flatten(feature_maps)


print("\n======================================")
print("             FLATTENED")
print("======================================")

print(flattened)

print("\nNumber of values:", len(flattened))