import numpy as np


def max_pooling2d(feature_map, pool_size=2, stride=2, return_cache=False):
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

    # Store where the maximum value came from
    max_positions = []

    output_row = 0

    for i in range(0, height - pool_size + 1, stride):

        output_col = 0

        for j in range(0, width - pool_size + 1, stride):

            region = feature_map[
                i:i + pool_size,
                j:j + pool_size
            ]

            # Find maximum
            max_value = np.max(region)

            pooled_map[output_row, output_col] = max_value

            # Find position of maximum inside the region
            max_index = np.unravel_index(
                np.argmax(region),
                region.shape
            )

            max_row = i + max_index[0]
            max_col = j + max_index[1]

            max_positions.append(
                (output_row, output_col, max_row, max_col)
            )

            output_col += 1

        output_row += 1

    if return_cache:
        cache = {
            "input_shape": feature_map.shape,
            "pool_size": pool_size,
            "stride": stride,
            "max_positions": max_positions
        }

        return pooled_map, cache

    return pooled_map


def max_pooling_backward(pooled_gradient, cache):
    pooled_gradient = np.array(pooled_gradient, dtype=float)

    input_shape = cache["input_shape"]
    max_positions = cache["max_positions"]

    height, width = input_shape

    # Gradient with same shape as original feature map
    input_gradient = np.zeros((height, width))

    for output_row, output_col, max_row, max_col in max_positions:

        # Gradient goes only to the maximum position
        input_gradient[max_row, max_col] += (
            pooled_gradient[output_row, output_col]
        )

    return input_gradient