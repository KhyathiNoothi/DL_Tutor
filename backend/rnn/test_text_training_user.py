from .text_training_model import train_text_dataset


# -----------------------------------------
# Get dataset size
# -----------------------------------------

num_samples = int(
    input("Enter number of training samples: ")
)


texts = []
targets = []


# -----------------------------------------
# Get training data
# -----------------------------------------

print("\nEnter training data:")

for i in range(num_samples):

    print(f"\nSample {i + 1}")

    text = input("Enter text: ")

    target = float(
        input("Enter target (0 or 1): ")
    )

    texts.append(text)
    targets.append(target)


# -----------------------------------------
# Training parameters
# -----------------------------------------

learning_rate = float(
    input("\nEnter learning rate: ")
)

epochs = int(
    input("Enter number of epochs: ")
)

hidden_size = int(
    input("Enter hidden size: ")
)

embedding_size = int(
    input("Enter embedding size: ")
)


# -----------------------------------------
# Train
# -----------------------------------------

result = train_text_dataset(
    texts=texts,
    targets=targets,
    learning_rate=learning_rate,
    epochs=epochs,
    hidden_size=hidden_size,
    embedding_size=embedding_size
)


# -----------------------------------------
# Display result
# -----------------------------------------

print("\n==============================")
print("TEXT RNN TRAINING RESULT")
print("==============================")


print("\nVocabulary:")
print(result["vocabulary"])


print("\nFinal Average Loss:")
print(result["final_loss"])


print("\nTraining Summary:")

for epoch in result["history"]:

    print(
        f"Epoch {epoch['epoch']} | "
        f"Average Loss: "
        f"{epoch['average_loss']:.6f}"
    )