from .tokenizer import tokenize
from .vocabulary import Vocabulary
from .embedding import Embedding


text = "I love this movie"

# -----------------------------
# Tokenization
# -----------------------------

tokens = tokenize(text)

print("Tokens:")
print(tokens)


# -----------------------------
# Vocabulary
# -----------------------------

vocab = Vocabulary()

vocab.build(tokens)

print("\nVocabulary:")
print(vocab.word_to_id)


# -----------------------------
# Convert words → IDs
# -----------------------------

token_ids = vocab.encode(tokens)

print("\nToken IDs:")
print(token_ids)


# -----------------------------
# Embedding
# -----------------------------

embedding_size = 3

embedding = Embedding(
    vocabulary_size=vocab.size(),
    embedding_size=embedding_size
)

vectors = embedding.forward(token_ids)

print("\nEmbedding vectors:")

for token, token_id, vector in zip(
    tokens,
    token_ids,
    vectors
):
    print(f"{token} ({token_id}) → {vector}")