import numpy as np


def sigmoid(x):
    """
    Convert a value into a probability between 0 and 1.
    Used for binary classification.
    """

    return 1 / (1 + np.exp(-x))


def softmax(logits):
    """
    Convert logits into probabilities.

    Used for multi-class classification.
    """

    logits = np.array(logits, dtype=float)

    # Numerical stability
    shifted_logits = logits - np.max(logits)

    exp_values = np.exp(shifted_logits)

    probabilities = exp_values / np.sum(exp_values)

    return probabilities


def predict_binary(logit):
    """
    Binary classification prediction.

    Returns:
        probability and predicted class
    """

    probability = sigmoid(logit)

    predicted_class = 1 if probability >= 0.5 else 0

    return {
        "probability": float(probability),
        "predicted_class": predicted_class
    }


def predict_multiclass(logits):
    """
    Multi-class classification prediction.

    Returns:
        probabilities and predicted class
    """

    probabilities = softmax(logits)

    predicted_class = int(np.argmax(probabilities))

    return {
        "probabilities": probabilities.tolist(),
        "predicted_class": predicted_class
    }