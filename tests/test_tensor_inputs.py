import unittest

import torch

from transformers_learning import build_input_ids, describe_input_ids


class TensorInputTests(unittest.TestCase):
    # Day 1.6: TL-001 verify token ids become a batch-first torch.long tensor.
    def test_build_input_ids_creates_expected_tensor(self) -> None:
        input_ids = build_input_ids([101, 2023, 2003, 102])

        self.assertEqual(input_ids.shape, torch.Size([1, 4]))
        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(input_ids.tolist(), [[101, 2023, 2003, 102]])

    # Day 1.7: TL-001 protect the input contract from empty token sequences.
    def test_build_input_ids_rejects_empty_sequence(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one token id"):
            build_input_ids([])

    # Day 1.8: TL-001 reject one-dimensional input because later transformer
    # layers expect batch and sequence dimensions.
    def test_describe_input_ids_requires_two_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "2D tensor"):
            describe_input_ids(torch.tensor([101, 102], dtype=torch.long))

    # Day 1.9: TL-001 reject floating-point token indices before embedding lookup.
    def test_describe_input_ids_requires_long_dtype(self) -> None:
        with self.assertRaisesRegex(TypeError, "torch.long"):
            describe_input_ids(torch.tensor([[101.0, 102.0]]))


if __name__ == "__main__":
    unittest.main()
