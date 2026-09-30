import numpy as np

from .tokenizer import tokenize


def predict_text(
    text,
    vocabulary,
    embedding,
    model,
    threshold=0.5
):
    """
    Predict the class of a new text using a trained RNN.

    Parameters:
        text        : input text from the user
        vocabulary : trained Vocabulary object
        embedding  : trained Embedding object
        model       : trained RNNModel
        threshold   : probability threshold for classification

    Returns:
        prediction details
    """

    if not isinstance(text, str):
        raise ValueError("Text must be a string")

    if len(text.strip()) == 0:
        raise ValueError("Text cannot be empty")

    if threshold <= 0 or threshold >= 1:
        raise ValueError("Threshold must be between 0 and 1")

    # --------------------------------
    # 1. Tokenize the text
    # --------------------------------

    tokens = tokenize(text)

    if len(tokens) == 0:
        raise ValueError("Text must contain at least one token")

    # --------------------------------
    # 2. Convert tokens to IDs
    # --------------------------------

    token_ids = vocabulary.encode(tokens)

    # --------------------------------
    # 3. Convert token IDs to embeddings
    # --------------------------------

    sequence = embedding.forward(token_ids)

    # --------------------------------
    # 4. Create initial hidden state
    # --------------------------------

    initial_hidden = np.zeros(model.hidden_size)

    # --------------------------------
    # 5. Run the trained RNN
    # --------------------------------

    result = model.predict(
        sequence=sequence,
        initial_hidden=initial_hidden
    )

    probability = float(result["probability"])

    # --------------------------------
    # 6. Convert probability to class
    # --------------------------------

    if probability >= threshold:
        prediction = 1
        label = "Positive"
    else:
        prediction = 0
        label = "Negative"

    return {
        "text": text,
        "tokens": tokens,
        "token_ids": token_ids,
        "probability": probability,
        "prediction": prediction,
        "label": label,
        "forward_history": result["forward_history"]
    }