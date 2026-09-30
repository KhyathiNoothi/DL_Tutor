class Vocabulary:

    def __init__(self):
        self.word_to_id = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        self.id_to_word = {
            0: "<PAD>",
            1: "<UNK>"
        }

    def build(self, tokens):
        """
        Build vocabulary from a list of tokens.
        """

        for token in tokens:

            if token not in self.word_to_id:

                new_id = len(self.word_to_id)

                self.word_to_id[token] = new_id
                self.id_to_word[new_id] = token

    def encode(self, tokens):
        """
        Convert words into integer IDs.
        """

        ids = []

        for token in tokens:

            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
            else:
                ids.append(self.word_to_id["<UNK>"])

        return ids

    def decode(self, ids):
        """
        Convert integer IDs back into words.
        """

        words = []

        for id_value in ids:

            if id_value in self.id_to_word:
                words.append(self.id_to_word[id_value])
            else:
                words.append("<UNK>")

        return words

    def size(self):
        return len(self.word_to_id)