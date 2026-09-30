import numpy as np

from .filters import apply_multiple_filters
from .activation import relu
from .pooling import max_pooling2d
from .flatten import flatten
from .dense import DenseLayer
from .prediction import predict_binary, predict_multiclass


class CNN:

    def __init__(
        self,
        kernels,
        pool_size=2,
        pool_stride=2,
        dense_output_size=1,
        dense_activation="linear"
    ):
        self.kernels = [
            np.array(kernel, dtype=float)
            for kernel in kernels
        ]

        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.dense_output_size = dense_output_size
        self.dense_activation = dense_activation

        self.dense = None
        self.dense_initialized = False

    def forward(self, image, stride=1, padding=0):

        image = np.array(image, dtype=float)

        # -----------------------------
        # 1. Convolution
        # -----------------------------

        feature_maps = apply_multiple_filters(
            image,
            self.kernels,
            stride,
            padding
        )

        # -----------------------------
        # 2. ReLU
        # -----------------------------

        activated_maps = []

        for feature_map in feature_maps:

            activated_map = relu(feature_map)

            activated_maps.append(activated_map)

        # -----------------------------
        # 3. Max Pooling
        # -----------------------------

        pooled_maps = []
        pooling_caches = []

        for activated_map in activated_maps:

            pooled_map, cache = max_pooling2d(
                activated_map,
                self.pool_size,
                self.pool_stride,
                return_cache=True
            )

            pooled_maps.append(pooled_map)
            pooling_caches.append(cache)

        # -----------------------------
        # 4. Flatten
        # -----------------------------

        flattened = flatten(pooled_maps)

        # -----------------------------
        # 5. Dense
        # -----------------------------

        if not self.dense_initialized:

            self.dense = DenseLayer(
                input_size=len(flattened),
                output_size=self.dense_output_size,
                activation=self.dense_activation
            )

            self.dense_initialized = True

        dense_result = self.dense.forward(flattened)

        return {
            "feature_maps": feature_maps,
            "activated_maps": activated_maps,
            "pooled_maps": pooled_maps,
            "pooling_caches": pooling_caches,
            "flattened": flattened,
            "dense": dense_result
        }

    def predict(self, dense_output, task="binary"):

        logits = dense_output["output"]

        if task == "binary":

            result = predict_binary(logits[0])

        elif task == "multiclass":

            result = predict_multiclass(logits)

        else:

            raise ValueError(
                "Task must be 'binary' or 'multiclass'"
            )

        return result