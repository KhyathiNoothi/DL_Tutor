from .text_training import train_text_rnn


text = input("Enter text: ")

target = float(
    input("Enter target (0 or 1): ")
)

learning_rate = float(
    input("Enter learning rate: ")
)

epochs = int(
    input("Enter number of epochs: ")
)


result = train_text_rnn(
    text=text,
    target=target,
    learning_rate=learning_rate,
    epochs=epochs
)


print("\n==============================")
print("TEXT RNN RESULT")
print("==============================")


print("\nOriginal Text:")
print(result["text"])


print("\nTokens:")
print(result["tokens"])


print("\nToken IDs:")
print(result["token_ids"])


print("\nVocabulary:")
print(result["vocabulary"])


print("\nEmbeddings:")

for token, vector in zip(
    result["tokens"],
    result["embeddings"]
):
    print(f"{token} → {vector}")


print("\nFinal Hidden State:")
print(result["final_hidden"])


print("\nFinal Probability:")
print(result["final_probability"])


print("\nFinal Loss:")
print(result["final_loss"])