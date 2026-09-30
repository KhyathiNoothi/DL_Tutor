from .text_training_model import train_text_dataset
from .inference import predict_text


# =====================================
# TRAINING DATA
# =====================================

num_samples = int(input("Enter number of training samples: "))

texts = []
targets = []

print("\nEnter training data:")

for i in range(num_samples):

    print(f"\nSample {i + 1}")

    text = input("Enter text: ")
    target = float(input("Enter target (0 or 1): "))

    texts.append(text)
    targets.append(target)


# =====================================
# TRAINING PARAMETERS
# =====================================

learning_rate = float(input("\nEnter learning rate: "))
epochs = int(input("Enter number of epochs: "))
hidden_size = int(input("Enter hidden size: "))
embedding_size = int(input("Enter embedding size: "))


# =====================================
# TRAIN
# =====================================

print("\n==============================")
print("TRAINING RNN")
print("==============================")

result = train_text_dataset(
    texts=texts,
    targets=targets,
    learning_rate=learning_rate,
    epochs=epochs,
    hidden_size=hidden_size,
    embedding_size=embedding_size
)


# =====================================
# TRAINING RESULT
# =====================================

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print("Final Loss:", result["final_loss"])

print("\nVocabulary:")
print(result["vocabulary"].word_to_id)


# =====================================
# INFERENCE
# =====================================

print("\n==============================")
print("RNN INFERENCE")
print("==============================")

new_text = input("\nEnter a new text to classify: ")


prediction_result = predict_text(
    text=new_text,
    vocabulary=result["vocabulary"],
    embedding=result["embedding"],
    model=result["model"]
)


# =====================================
# DISPLAY RESULT
# =====================================

print("\n==============================")
print("PREDICTION RESULT")
print("==============================")

print("Text:")
print(prediction_result["text"])

print("\nTokens:")
print(prediction_result["tokens"])

print("\nToken IDs:")
print(prediction_result["token_ids"])

print("\nProbability:")
print(prediction_result["probability"])

print("\nPrediction:")
print(prediction_result["prediction"])

print("\nLabel:")
print(prediction_result["label"])