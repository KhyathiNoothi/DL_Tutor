import re


def tokenize(text):
    """
    Convert text into a list of lowercase words.
    """

    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # Convert to lowercase
    text = text.lower()

    # Extract words
    tokens = re.findall(r"\b\w+\b", text)

    return tokens