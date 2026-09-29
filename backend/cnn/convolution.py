import numpy as np


def convolution2d(image, kernel, stride=1, padding=0):
    """
    Perform 2D convolution with stride and padding.

    Parameters:
        image: 2D input image
        kernel: 2D filter
        stride: number of pixels the kernel moves
        padding: number of zero-padding layers around image

    Returns:
        2D feature map
    """

    image = np.array(image, dtype=float)
    kernel = np.array(kernel, dtype=float)

    # --------------------------------------
    # Validate stride and padding
    # --------------------------------------

    if stride <= 0:
        raise ValueError("Stride must be greater than 0")

    if padding < 0:
        raise ValueError("Padding cannot be negative")

    # --------------------------------------
    # Apply padding
    # --------------------------------------

    if padding > 0:
        image = np.pad(
            image,
            (
                (padding, padding),
                (padding, padding)
            ),
            mode="constant",
            constant_values=0
        )

    # --------------------------------------
    # Get dimensions
    # --------------------------------------

    image_height, image_width = image.shape

    kernel_height, kernel_width = kernel.shape

    # --------------------------------------
    # Check kernel size
    # --------------------------------------

    if (
        kernel_height > image_height
        or kernel_width > image_width
    ):
        raise ValueError(
            "Kernel cannot be larger than the image"
        )

    # --------------------------------------
    # Calculate output dimensions
    # --------------------------------------

    output_height = (
        (image_height - kernel_height)
        // stride
    ) + 1

    output_width = (
        (image_width - kernel_width)
        // stride
    ) + 1

    feature_map = np.zeros(
        (output_height, output_width)
    )

    # --------------------------------------
    # Perform convolution
    # --------------------------------------

    output_row = 0

    for i in range(
        0,
        image_height - kernel_height + 1,
        stride
    ):

        output_col = 0

        for j in range(
            0,
            image_width - kernel_width + 1,
            stride
        ):

            region = image[
                i:i + kernel_height,
                j:j + kernel_width
            ]

            feature_map[
                output_row,
                output_col
            ] = np.sum(
                region * kernel
            )

            output_col += 1

        output_row += 1

    return feature_map
def convolution_backward(image, kernel, upstream_gradient, stride=1, padding=0):
    image = np.array(image, dtype=float)
    kernel = np.array(kernel, dtype=float)
    upstream_gradient = np.array(upstream_gradient, dtype=float)

    if stride <= 0:
        raise ValueError("Stride must be greater than 0")

    if padding < 0:
        raise ValueError("Padding cannot be negative")

    # Apply the same padding used during forward pass
    if padding > 0:
        padded_image = np.pad(
            image,
            ((padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0
        )
    else:
        padded_image = image.copy()

    image_height, image_width = padded_image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = ((image_height - kernel_height) // stride) + 1
    output_width = ((image_width - kernel_width) // stride) + 1

    if upstream_gradient.shape != (output_height, output_width):
        raise ValueError(
            "Upstream gradient shape does not match convolution output"
        )

    # Gradient for the kernel/filter
    kernel_gradient = np.zeros_like(kernel)

    # Gradient for the padded input
    input_gradient = np.zeros_like(padded_image)

    output_row = 0

    for i in range(0, image_height - kernel_height + 1, stride):

        output_col = 0

        for j in range(0, image_width - kernel_width + 1, stride):

            # Same region used during forward convolution
            region = padded_image[
                i:i + kernel_height,
                j:j + kernel_width
            ]

            gradient = upstream_gradient[
                output_row,
                output_col
            ]

            # Gradient with respect to kernel
            kernel_gradient += region * gradient

            # Gradient with respect to input
            input_gradient[
                i:i + kernel_height,
                j:j + kernel_width
            ] += kernel * gradient

            output_col += 1

        output_row += 1

    # Remove padding from input gradient
    if padding > 0:
        input_gradient = input_gradient[
            padding:-padding,
            padding:-padding
        ]

    return {
        "kernel_gradient": kernel_gradient,
        "input_gradient": input_gradient
    }