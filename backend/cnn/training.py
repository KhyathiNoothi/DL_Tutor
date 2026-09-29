import numpy as np

from loss import binary_cross_entropy
from activation import relu_backward
from pooling import max_pooling_backward
from convolution import convolution_backward
from optimizer import update_parameters, update_kernel


def train_cnn(
    cnn,
    image,
    target,
    learning_rate,
    epochs,
    stride=1,
    padding=0
):

    image = np.array(image, dtype=float)
    target = float(target)

    if target not in [0, 1]:
        raise ValueError("Binary classification target must be 0 or 1")

    if learning_rate <= 0:
        raise ValueError("Learning rate must be greater than 0")

    if epochs <= 0:
        raise ValueError("Epochs must be greater than 0")

    history = []

    for epoch in range(epochs):

        # ==================================================
        # 1. FORWARD PASS
        # ==================================================

        forward_result = cnn.forward(
            image,
            stride,
            padding
        )

        # Save forward values BEFORE updating anything
        feature_maps = [
            feature_map.copy()
            for feature_map in forward_result["feature_maps"]
        ]

        activated_maps = [
            activated_map.copy()
            for activated_map in forward_result["activated_maps"]
        ]

        pooled_maps = [
            pooled_map.copy()
            for pooled_map in forward_result["pooled_maps"]
        ]

        flattened = np.array(
            forward_result["flattened"],
            dtype=float
        ).copy()

        dense_result = forward_result["dense"]

        # ==================================================
        # 2. PREDICTION
        # ==================================================

        logit = float(
            dense_result["output"][0]
        )

        probability = 1 / (1 + np.exp(-logit))

        # ==================================================
        # 3. LOSS
        # ==================================================

        loss = binary_cross_entropy(
            [target],
            [probability]
        )

        # ==================================================
        # 4. BACKPROPAGATION
        # ==================================================

        # Sigmoid + Binary Cross Entropy:
        #
        # dL/dz = prediction - target

        dL_dz = probability - target

        # ==================================================
        # 5. DENSE GRADIENTS
        # ==================================================

        dense_weights = cnn.dense.weights.copy()

        dense_weight_gradient = (
            flattened.reshape(-1, 1) * dL_dz
        )

        dense_bias_gradient = np.array(
            [dL_dz]
        )

        # Gradient flowing back to flattened layer
        flattened_gradient = (
            dense_weights[:, 0] * dL_dz
        )

        # ==================================================
        # 6. FLATTEN BACKWARD
        # ==================================================

        pooled_gradients = []

        position = 0

        for pooled_map in pooled_maps:

            size = pooled_map.size

            gradient_part = flattened_gradient[
                position:position + size
            ]

            gradient_part = gradient_part.reshape(
                pooled_map.shape
            )

            pooled_gradients.append(
                gradient_part
            )

            position += size

        # ==================================================
        # 7. POOLING BACKWARD
        # ==================================================

        activated_gradients = []

        for pooled_gradient, cache in zip(
            pooled_gradients,
            forward_result["pooling_caches"]
        ):

            gradient = max_pooling_backward(
                pooled_gradient,
                cache
            )

            activated_gradients.append(
                gradient
            )

        # ==================================================
        # 8. RELU BACKWARD
        # ==================================================

        feature_map_gradients = []

        for feature_map, upstream_gradient in zip(
            feature_maps,
            activated_gradients
        ):

            gradient = relu_backward(
                feature_map,
                upstream_gradient
            )

            feature_map_gradients.append(
                gradient
            )

        # ==================================================
        # 9. CONVOLUTION BACKWARD
        # ==================================================

        kernel_gradients = []

        input_gradients = []

        for kernel, feature_map_gradient in zip(
            cnn.kernels,
            feature_map_gradients
        ):

            result = convolution_backward(
                image,
                kernel,
                feature_map_gradient,
                stride,
                padding
            )

            kernel_gradients.append(
                result["kernel_gradient"]
            )

            input_gradients.append(
                result["input_gradient"]
            )

        # ==================================================
        # 10. SAVE OLD PARAMETERS
        # ==================================================

        old_dense_weights = (
            cnn.dense.weights.copy()
        )

        old_dense_biases = (
            cnn.dense.biases.copy()
        )

        old_kernels = [
            kernel.copy()
            for kernel in cnn.kernels
        ]

        # ==================================================
        # 11. UPDATE DENSE PARAMETERS
        # ==================================================

        dense_update = update_parameters(
            cnn.dense.weights,
            cnn.dense.biases,
            dense_weight_gradient,
            dense_bias_gradient,
            learning_rate
        )

        cnn.dense.weights = (
            dense_update["weights"]
        )

        cnn.dense.biases = (
            dense_update["biases"]
        )

        # ==================================================
        # 12. UPDATE CNN FILTERS
        # ==================================================

        updated_kernels = []

        for kernel, kernel_gradient in zip(
            cnn.kernels,
            kernel_gradients
        ):

            new_kernel = update_kernel(
                kernel,
                kernel_gradient,
                learning_rate
            )

            updated_kernels.append(
                new_kernel
            )

        cnn.kernels = updated_kernels

        # ==================================================
        # 13. SAVE COMPLETE EPOCH HISTORY
        # ==================================================

        history.append({

            # Basic information
            "epoch": epoch + 1,

            # Forward pass
            "feature_maps": [
                feature_map.tolist()
                for feature_map in feature_maps
            ],

            "activated_maps": [
                activated_map.tolist()
                for activated_map in activated_maps
            ],

            "pooled_maps": [
                pooled_map.tolist()
                for pooled_map in pooled_maps
            ],

            "flattened": flattened.tolist(),

            # Prediction
            "logit": float(logit),

            "probability": float(probability),

            # Loss
            "target": float(target),

            "loss": float(loss),

            # Backward pass
            "dense_weight_gradient":
                dense_weight_gradient.tolist(),

            "dense_bias_gradient":
                dense_bias_gradient.tolist(),

            "flattened_gradient":
                flattened_gradient.tolist(),

            "pooled_gradients": [
                gradient.tolist()
                for gradient in pooled_gradients
            ],

            "activated_gradients": [
                gradient.tolist()
                for gradient in activated_gradients
            ],

            "feature_map_gradients": [
                gradient.tolist()
                for gradient in feature_map_gradients
            ],

            "kernel_gradients": [
                gradient.tolist()
                for gradient in kernel_gradients
            ],

            # Parameters BEFORE update
            "old_dense_weights":
                old_dense_weights.tolist(),

            "old_dense_biases":
                old_dense_biases.tolist(),

            "old_kernels": [
                kernel.tolist()
                for kernel in old_kernels
            ],

            # Parameters AFTER update
            "new_dense_weights":
                cnn.dense.weights.tolist(),

            "new_dense_biases":
                cnn.dense.biases.tolist(),

            "new_kernels": [
                kernel.tolist()
                for kernel in cnn.kernels
            ]
        })

        # ==================================================
        # 14. PRINT EPOCH SUMMARY
        # ==================================================

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Probability: {probability:.6f} | "
            f"Loss: {loss:.6f}"
        )

    return {

        "final_probability":
            float(probability),

        "final_loss":
            float(loss),

        "history":
            history
    }