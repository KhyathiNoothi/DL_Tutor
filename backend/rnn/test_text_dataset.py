from .text_dataset import TextDataset


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


dataset = TextDataset(
    texts=texts,
    targets=targets
)


print("Dataset size:")
print(dataset.size())


print("\nAll samples:")

for sample in dataset.all_samples():
    print(
        f"Text: {sample['text']} "
        f"| Target: {sample['target']}"
    )


print("\nFirst sample:")

print(dataset.get_sample(0))