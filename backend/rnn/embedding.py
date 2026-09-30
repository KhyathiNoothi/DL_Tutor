import numpy as np


class Embedding:

    def __init__(self, vocabulary_size, embedding_size):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        # Random embedding matrix
        self.weights = (
            np.random.randn(vocabulary_size, embedding_size) * 0.01
        )

    def forward(self, token_ids):
        """
        Convert token IDs into embedding vectors.
        """

        token_ids = np.array(token_ids, dtype=int)

        if np.any(token_ids < 0):
            raise ValueError("Token IDs cannot be negative")

        if np.any(token_ids >= self.vocabulary_size):
            raise ValueError(
                "Token ID is outside vocabulary range"
            )

        return self.weights[token_ids]