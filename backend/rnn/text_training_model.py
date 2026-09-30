import numpy as np

from .tokenizer import tokenize
from .vocabulary import Vocabulary
from .embedding import Embedding
from .model import RNNModel
from .loss import binary_cross_entropy
from .backprop import bptt
from .optimizer import update_parameters


def train_text_dataset(
    texts,
    targets,
    learning_rate,
    epochs,
    hidden_size=4,
    embedding_size=4
):
    """
    Train one persistent RNN model on multiple text samples.
    """

    # --------------------------------
    # Validation
    # --------------------------------

    if len(texts) != len(targets):
        raise ValueError(
            "Number of texts must match number of targets"
        )

    if len(texts) == 0:
        raise ValueError(
            "Dataset cannot be empty"
        )

    if learning_rate <= 0:
        raise ValueError(
            "Learning rate must be greater than 0"
        )

    if epochs <= 0:
        raise ValueError(
            "Epochs must be greater than 0"
        )

    # --------------------------------
    # Build vocabulary
    # --------------------------------

    vocabulary = Vocabulary()

    tokenized_texts = []

    for text in texts:

        tokens = tokenize(text)

        if len(tokens) == 0:
            raise ValueError(
                "A text sample cannot be empty"
            )

        tokenized_texts.append(tokens)

        vocabulary.build(tokens)

    # --------------------------------
    # Create embedding
    # --------------------------------

    embedding = Embedding(
        vocabulary_size=vocabulary.size(),
        embedding_size=embedding_size
    )

    # --------------------------------
    # Create ONE persistent RNN model
    # --------------------------------

    model = RNNModel(
        input_size=embedding_size,
        hidden_size=hidden_size
    )

    # --------------------------------
    # Initial hidden state
    # --------------------------------

    initial_hidden = np.zeros(hidden_size)

    history = []

    # =================================
    # EPOCH LOOP
    # =================================

    for epoch in range(epochs):

        epoch_loss = 0.0

        sample_history = []

        # -----------------------------
        # Process every sentence
        # -----------------------------

        for index in range(len(texts)):

            text = texts[index]
            target = float(targets[index])

            tokens = tokenized_texts[index]

            # -------------------------
            # Words → IDs
            # -------------------------

            token_ids = vocabulary.encode(tokens)

            # -------------------------
            # IDs → Embeddings
            # -------------------------

            sequence = embedding.forward(token_ids)

            # -------------------------
            # Forward pass
            # -------------------------

            forward_result = model.forward(
                sequence=sequence,
                initial_hidden=initial_hidden
            )

            final_hidden = np.array(
                forward_result["final_hidden"],
                dtype=float
            )

            # -------------------------
            # Output layer
            # -------------------------

            output_result = model.output_layer.forward(
                final_hidden
            )

            probability = output_result["probability"]
            logit = output_result["logit"]

            # -------------------------
            # Loss
            # -------------------------

            loss = binary_cross_entropy(
                [target],
                [probability]
            )

            # -------------------------
            # BPTT
            # -------------------------

            gradients = bptt(
                cell=model.cell,
                output_layer=model.output_layer,
                sequence=sequence,
                forward_history=forward_result["history"],
                target=target,
                probability=probability
            )

            # -------------------------
            # Save old parameters
            # -------------------------

            old_Wxh = model.cell.Wxh.copy()
            old_Whh = model.cell.Whh.copy()
            old_bh = model.cell.bh.copy()

            old_Why = model.output_layer.weights.copy()
            old_by = model.output_layer.bias

            # -------------------------
            # Gradient descent
            # -------------------------

            updated_parameters = update_parameters(
                Wxh=model.cell.Wxh,
                Whh=model.cell.Whh,
                bh=model.cell.bh,
                Why=model.output_layer.weights,
                by=model.output_layer.bias,

                dL_dWxh=gradients["dL_dWxh"],
                dL_dWhh=gradients["dL_dWhh"],
                dL_dbh=gradients["dL_dbh"],
                dL_dWhy=gradients["dL_dWhy"],
                dL_dby=gradients["dL_dby"],

                learning_rate=learning_rate
            )

            # -------------------------
            # Update SAME model
            # -------------------------

            model.cell.Wxh = updated_parameters["Wxh"]
            model.cell.Whh = updated_parameters["Whh"]
            model.cell.bh = updated_parameters["bh"]

            model.output_layer.weights = (
                updated_parameters["Why"]
            )

            model.output_layer.bias = (
                updated_parameters["by"]
            )

            epoch_loss += loss

            sample_history.append({
                "text": text,
                "tokens": tokens,
                "token_ids": token_ids,
                "target": target,
                "logit": float(logit),
                "probability": float(probability),
                "loss": float(loss),

                "old_parameters": {
                    "Wxh": old_Wxh.tolist(),
                    "Whh": old_Whh.tolist(),
                    "bh": old_bh.tolist(),
                    "Why": old_Why.tolist(),
                    "by": float(old_by)
                },

                "new_parameters": {
                    "Wxh": model.cell.Wxh.tolist(),
                    "Whh": model.cell.Whh.tolist(),
                    "bh": model.cell.bh.tolist(),
                    "Why": model.output_layer.weights.tolist(),
                    "by": float(model.output_layer.bias)
                }
            })

        # --------------------------------
        # Average epoch loss
        # --------------------------------

        average_loss = epoch_loss / len(texts)

        history.append({
            "epoch": epoch + 1,
            "average_loss": float(average_loss),
            "samples": sample_history
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Average Loss: {average_loss:.6f}"
        )

    return {
        "vocabulary": vocabulary.word_to_id,
        "final_loss": float(average_loss),
        "model": model,
        "embedding": embedding,
        "history": history
    }