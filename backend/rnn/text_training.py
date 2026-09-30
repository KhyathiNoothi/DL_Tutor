import numpy as np

from .tokenizer import tokenize
from .vocabulary import Vocabulary
from .embedding import Embedding
from .cell import RNNCell
from .training import train_rnn


def train_text_rnn(
    text,
    target,
    learning_rate,
    epochs,
    hidden_size=4,
    embedding_size=4
):
    """
    Train the RNN using text input.

    Flow:

    Text
    ↓
    Tokenization
    ↓
    Vocabulary
    ↓
    Token IDs
    ↓
    Embeddings
    ↓
    RNN
    ↓
    Prediction
    """

    # -----------------------------
    # 1. Tokenize text
    # -----------------------------

    tokens = tokenize(text)

    if len(tokens) == 0:
        raise ValueError("Text cannot be empty")

    # -----------------------------
    # 2. Build vocabulary
    # -----------------------------

    vocabulary = Vocabulary()

    vocabulary.build(tokens)

    # -----------------------------
    # 3. Convert words → IDs
    # -----------------------------

    token_ids = vocabulary.encode(tokens)

    # -----------------------------
    # 4. Create embedding layer
    # -----------------------------

    embedding = Embedding(
        vocabulary_size=vocabulary.size(),
        embedding_size=embedding_size
    )

    # -----------------------------
    # 5. Convert IDs → vectors
    # -----------------------------

    sequence = embedding.forward(token_ids)

    # -----------------------------
    # 6. Create RNN
    # -----------------------------

    input_size = embedding_size

    cell = RNNCell(
        input_size=input_size,
        hidden_size=hidden_size
    )

    # Initial hidden state
    initial_hidden = np.zeros(hidden_size)

    # -----------------------------
    # 7. Train RNN
    # -----------------------------

    result = train_rnn(
        cell=cell,
        sequence=sequence,
        initial_hidden=initial_hidden,
        target=target,
        learning_rate=learning_rate,
        epochs=epochs
    )

    # -----------------------------
    # 8. Return everything
    # -----------------------------

    return {
        "text": text,
        "tokens": tokens,
        "token_ids": token_ids,
        "embeddings": sequence.tolist(),
        "vocabulary": vocabulary.word_to_id,
        "final_probability": result["final_probability"],
        "final_loss": result["final_loss"],
        "final_hidden": result["final_hidden"],
        "history": result["history"]
    }