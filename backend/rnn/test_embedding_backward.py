import numpy as np

from .embedding import Embedding


# -----------------------------------------
# Create embedding
# -----------------------------------------

embedding = Embedding(
    vocabulary_size=6,
    embedding_size=3
)


# -----------------------------------------
# Token IDs
# -----------------------------------------

token_ids = np.array([
    2,
    3,
    4
])


# -----------------------------------------
# Input gradients from RNN
# -----------------------------------------

dL_dinputs = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
])


# -----------------------------------------
# Save old embeddings
# -----------------------------------------

old_weights = embedding.weights.copy()


# -----------------------------------------
# Learning rate
# -----------------------------------------

learning_rate = 0.1


# -----------------------------------------
# Backward
# -----------------------------------------

gradients = embedding.backward(
    token_ids=token_ids,
    dL_dinputs=dL_dinputs,
    learning_rate=learning_rate
)


# -----------------------------------------
# Display
# -----------------------------------------

print("OLD EMBEDDING WEIGHTS:")
print(old_weights)


print("\nEMBEDDING GRADIENTS:")
print(gradients)


print("\nNEW EMBEDDING WEIGHTS:")
print(embedding.weights)


print("\nUpdated rows:")

for token_id in token_ids:
    print(
        f"Token ID {token_id}:"
    )
    print(
        embedding.weights[token_id]
    )