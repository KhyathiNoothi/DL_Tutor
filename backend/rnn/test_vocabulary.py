from .tokenizer import tokenize
from .vocabulary import Vocabulary


text = "I love this movie"

tokens = tokenize(text)

print("Tokens:")
print(tokens)


vocab = Vocabulary()

vocab.build(tokens)

print("\nWord → ID:")
print(vocab.word_to_id)

ids = vocab.encode(tokens)

print("\nEncoded IDs:")
print(ids)

decoded = vocab.decode(ids)

print("\nDecoded words:")
print(decoded)

print("\nVocabulary size:")
print(vocab.size())