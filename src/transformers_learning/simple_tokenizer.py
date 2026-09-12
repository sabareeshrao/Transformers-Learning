"""A deliberately small tokenizer for learning the text-to-id boundary."""

from __future__ import annotations

import torch

from .tensor_inputs import build_input_ids


class SimpleTokenizer:
    """Lowercase whitespace tokenizer backed by an explicit vocabulary."""

    # Day 2.2: TL-002 require an explicit token-to-id vocabulary and a known
    # fallback token instead of inventing ids for unseen text.
    def __init__(self, vocab: dict[str, int], unk_token: str = "[UNK]") -> None:
        if unk_token not in vocab:
            raise ValueError("unk_token must exist in the vocabulary")

        self.vocab = dict(vocab)
        self.unk_token = unk_token
        self.unk_token_id = self.vocab[unk_token]

    # Day 2.3: TL-002 introduce the smallest deterministic text normalization:
    # lowercase the text and split on whitespace only.
    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    # Day 2.4: TL-002 translate known tokens through the vocabulary and map
    # unseen tokens to the configured unknown-token id.
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    # Day 2.5: TL-002 connect tokenization and vocabulary lookup into one
    # reusable text-to-id operation.
    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)

        if not tokens:
            raise ValueError("text must produce at least one token")

        return self.convert_tokens_to_ids(tokens)

    # Day 2.6: TL-002 connect today's text encoding to the Day 1 batch-first
    # tensor contract instead of creating a second tensor representation.
    def encode_to_tensor(self, text: str) -> torch.Tensor:
        return build_input_ids(self.encode(text))
