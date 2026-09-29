import numpy as np


def max_pooling2d(feature_map, pool_size=2, stride=2):
    """
    Perform 2D max pooling.

    Parameters:
        feature_map: 2D input feature map
        pool_size: size of pooling window
        stride: number of pixels the window moves

    Returns:
        2D pooled feature map
    """

    feature_map = np.array(feature_map, dtype=float)

    if pool_size <= 0:
        raise ValueError("Pool size must be greater than 0")

    if stride <= 0:
        raise ValueError("Stride must be greater than 0")

    height, width = feature_map.shape

    if pool_size > height or pool_size > width:
        raise ValueError("Pool size cannot be larger than the feature map")

    output_height = ((height - pool_size) // stride) + 1
    output_width = ((width - pool_size) // stride) + 1

    pooled_map = np.zeros((output_height, output_width))

    output_row = 0

    for i in range(0, height - pool_size + 1, stride):

        output_col = 0

        for j in range(0, width - pool_size + 1, stride):

            region = feature_map[
                i:i + pool_size,
                j:j + pool_size
            ]

            pooled_map[output_row, output_col] = np.max(region)

            output_col += 1

        output_row += 1

    return pooled_map