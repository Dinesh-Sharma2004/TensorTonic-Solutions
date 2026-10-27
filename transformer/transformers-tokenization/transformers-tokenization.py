class SimpleTokenizer:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }

        self.id_to_word = {
            idx: word for word, idx in self.word_to_id.items()
        }

        self.vocab_size = len(self.word_to_id)

    def build_vocab(self, texts: list[str]) -> None:
        """Builds the vocabulary in place."""
        unique_words = set()

        for text in texts:
            unique_words.update(text.lower().split())

        # Do not add special tokens again as regular vocabulary words.
        special_tokens = {
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        }
        unique_words.difference_update(special_tokens)

        # Sort alphabetically for deterministic ID assignments.
        sorted_words = sorted(unique_words)

        # Reset mappings while preserving the special-token IDs.
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }

        for word in sorted_words:
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx

        self.id_to_word = {
            idx: word for word, idx in self.word_to_id.items()
        }

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> list[int]:
        """Returns token IDs for the input text."""
        unk_id = self.word_to_id[self.unk_token]

        return [
            self.word_to_id.get(word, unk_id)
            for word in text.lower().split()
        ]

    def decode(self, ids: list[int]) -> str:
        """Returns the decoded, space-separated text."""
        return " ".join(
            self.id_to_word.get(idx, self.unk_token)
            for idx in ids
        )