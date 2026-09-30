import numpy as np

from .tokenizer import tokenize
from .vocabulary import Vocabulary
from .embedding import Embedding
from .cell import RNNCell
from .training import train_rnn


def train_text_dataset(
    texts,
    targets,
    learning_rate,
    epochs,
    hidden_size=4,
    embedding_size=4
):
    """
    Train an RNN using multiple text samples.

    Flow:

    Text dataset
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
        ↓
    Loss
        ↓
    BPTT
        ↓
    Parameter Update
    """

    if len(texts) != len(targets):
        raise ValueError(
            "Number of texts must match number of targets"
        )

    if len(texts) == 0:
        raise ValueError(
            "Dataset cannot be empty"
        )

    # --------------------------------
    # 1. Build vocabulary
    # --------------------------------

    vocabulary = Vocabulary()

    for text in texts:

        tokens = tokenize(text)

        if len(tokens) == 0:
            raise ValueError(
                "A text sample cannot be empty"
            )

        vocabulary.build(tokens)

    # --------------------------------
    # 2. Create embedding layer
    # --------------------------------

    embedding = Embedding(
        vocabulary_size=vocabulary.size(),
        embedding_size=embedding_size
    )

    # --------------------------------
    # 3. Create RNN
    # --------------------------------

    cell = RNNCell(
        input_size=embedding_size,
        hidden_size=hidden_size
    )

    # --------------------------------
    # 4. Initial hidden state
    # --------------------------------

    initial_hidden = np.zeros(hidden_size)

    history = []

    # --------------------------------
    # 5. Training
    # --------------------------------

    for epoch in range(epochs):

        epoch_loss = 0.0

        sample_history = []

        for text, target in zip(texts, targets):

            # Tokenize
            tokens = tokenize(text)

            # Convert words → IDs
            token_ids = vocabulary.encode(tokens)

            # IDs → vectors
            sequence = embedding.forward(token_ids)

            # Train one sample
            result = train_rnn(
                cell=cell,
                sequence=sequence,
                initial_hidden=initial_hidden,
                target=target,
                learning_rate=learning_rate,
                epochs=1
            )

            loss = result["final_loss"]
            probability = result["final_probability"]

            epoch_loss += loss

            sample_history.append({
                "text": text,
                "tokens": tokens,
                "token_ids": token_ids,
                "target": target,
                "probability": probability,
                "loss": loss
            })

        average_loss = epoch_loss / len(texts)

        history.append({
            "epoch": epoch + 1,
            "average_loss": average_loss,
            "samples": sample_history
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Average Loss: {average_loss:.6f}"
        )

    return {
        "vocabulary": vocabulary.word_to_id,
        "final_loss": average_loss,
        "history": history
    }