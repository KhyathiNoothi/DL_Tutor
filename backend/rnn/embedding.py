import numpy as np


class Embedding:

    def __init__(self, vocabulary_size, embedding_size):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        # Embedding matrix
        #
        # Each row represents one token.
        #
        # Example:
        # row 2 → embedding for "i"
        # row 3 → embedding for "love"
        #
        self.weights = (
            np.random.randn(
                vocabulary_size,
                embedding_size
            ) * 0.01
        )

    def forward(self, token_ids):
        """
        Convert token IDs into embedding vectors.
        """

        token_ids = np.array(
            token_ids,
            dtype=int
        )

        if np.any(token_ids < 0):
            raise ValueError(
                "Token IDs cannot be negative"
            )

        if np.any(
            token_ids >= self.vocabulary_size
        ):
            raise ValueError(
                "Token ID is outside vocabulary range"
            )

        return self.weights[token_ids]

    def backward(
        self,
        token_ids,
        dL_dinputs,
        learning_rate
    ):
        """
        Update the embedding vectors using
        gradients received from the RNN.
        """

        token_ids = np.array(
            token_ids,
            dtype=int
        )

        dL_dinputs = np.array(
            dL_dinputs,
            dtype=float
        )

        if len(token_ids) != len(dL_dinputs):
            raise ValueError(
                "Number of token IDs must match "
                "number of input gradients"
            )

        if learning_rate <= 0:
            raise ValueError(
                "Learning rate must be greater than 0"
            )

        # Store the gradients for the
        # complete embedding matrix.
        dL_dweights = np.zeros_like(
            self.weights
        )

        # Each token receives the gradient
        # corresponding to its position.
        for time_step, token_id in enumerate(
            token_ids
        ):

            dL_dweights[token_id] += (
                dL_dinputs[time_step]
            )

        # Gradient descent update
        self.weights -= (
            learning_rate * dL_dweights
        )

        return dL_dweights