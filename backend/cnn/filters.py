import numpy as np
from .convolution import convolution2d


def apply_multiple_filters(image, kernels, stride=1, padding=0):
    """
    Apply multiple filters to the same image.

    Parameters:
        image: 2D input image
        kernels: list of 2D filters
        stride: number of pixels each filter moves
        padding: number of zero-padding layers

    Returns:
        List of feature maps
    """

    image = np.array(image, dtype=float)

    if len(kernels) == 0:
        raise ValueError("At least one filter is required")

    feature_maps = []

    for kernel in kernels:

        kernel = np.array(kernel, dtype=float)

        feature_map = convolution2d(
            image,
            kernel,
            stride,
            padding
        )

        feature_maps.append(feature_map)

    return feature_maps