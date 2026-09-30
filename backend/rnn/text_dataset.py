class TextDataset:

    def __init__(self, texts, targets):
        if len(texts) != len(targets):
            raise ValueError(
                "Number of texts must match number of targets"
            )

        if len(texts) == 0:
            raise ValueError("Dataset cannot be empty")

        self.texts = texts
        self.targets = targets

    def size(self):
        return len(self.texts)

    def get_sample(self, index):
        return {
            "text": self.texts[index],
            "target": self.targets[index]
        }

    def all_samples(self):
        return [
            {
                "text": text,
                "target": target
            }
            for text, target in zip(
                self.texts,
                self.targets
            )
        ]