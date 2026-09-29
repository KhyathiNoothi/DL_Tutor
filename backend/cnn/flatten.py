import numpy as np


def flatten(feature_maps):
    """
    Flatten CNN feature maps into one 1D vector.
    """

    # If a single feature map is provided
    if isinstance(feature_maps, np.ndarray):
        return feature_maps.flatten()

    # If multiple feature maps are provided
    flattened_maps = []

    for feature_map in feature_maps:
        feature_map = np.array(feature_map, dtype=float)
        flattened_maps.extend(feature_map.flatten())

    return np.array(flattened_maps)