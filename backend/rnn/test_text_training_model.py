from .text_training_model import train_text_dataset


texts = [
    "I love this movie",
    "This movie is amazing",
    "I hate this movie",
    "This movie is terrible"
]

targets = [
    1,
    1,
    0,
    0
]


learning_rate = 0.1
epochs = 5


result = train_text_dataset(
    texts=texts,
    targets=targets,
    learning_rate=learning_rate,
    epochs=epochs
)


print("\n==============================")
print("TEXT RNN TRAINING RESULT")
print("==============================")


print("\nVocabulary:")
print(result["vocabulary"])


print("\nFinal Average Loss:")
print(result["final_loss"])


print("\nTraining History:")

for epoch in result["history"]:

    print(
        f"\nEpoch {epoch['epoch']}"
    )

    for sample in epoch["samples"]:

        print(
            f"Text: {sample['text']} | "
            f"Target: {sample['target']} | "
            f"Probability: {sample['probability']:.6f} | "
            f"Loss: {sample['loss']:.6f}"
        )