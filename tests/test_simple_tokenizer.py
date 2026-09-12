import unittest

import torch

from transformers_learning import SimpleTokenizer


class SimpleTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = SimpleTokenizer(
            {
                "[UNK]": 0,
                "transformers": 1,
                "turn": 2,
                "text": 3,
                "into": 4,
                "tokens": 5,
            }
        )

    # Day 2.8: TL-002 verify the first tokenizer performs only the normalization
    # we deliberately chose today: lowercase plus whitespace splitting.
    def test_tokenize_lowercases_and_splits_whitespace(self) -> None:
        self.assertEqual(
            self.tokenizer.tokenize("Transformers   Turn TEXT"),
            ["transformers", "turn", "text"],
        )

    # Day 2.9: TL-002 verify vocabulary lookup produces stable numeric token ids.
    def test_convert_tokens_to_ids_uses_vocabulary(self) -> None:
        self.assertEqual(
            self.tokenizer.convert_tokens_to_ids(["transformers", "tokens"]),
            [1, 5],
        )

    # Day 2.10: TL-002 keep unseen text representable through a stable unknown id.
    def test_unknown_token_uses_unk_id(self) -> None:
        self.assertEqual(
            self.tokenizer.convert_tokens_to_ids(["transformers", "attention"]),
            [1, 0],
        )

    # Day 2.11: TL-002 verify the complete Day 2 path reuses Day 1's tensor
    # contract: raw text -> tokens -> ids -> [batch, sequence] torch.long tensor.
    def test_encode_to_tensor_connects_to_day_1_contract(self) -> None:
        input_ids = self.tokenizer.encode_to_tensor("Transformers turn text into tokens")

        self.assertEqual(input_ids.shape, torch.Size([1, 5]))
        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(input_ids.tolist(), [[1, 2, 3, 4, 5]])

    # Day 2.12: TL-002 reject empty text before it reaches later model code.
    def test_encode_rejects_empty_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one token"):
            self.tokenizer.encode("   ")

    # Day 2.13: TL-002 fail fast when the configured unknown token has no id.
    def test_unknown_token_must_exist_in_vocabulary(self) -> None:
        with self.assertRaisesRegex(ValueError, "unk_token"):
            SimpleTokenizer({"transformers": 1})


if __name__ == "__main__":
    unittest.main()
