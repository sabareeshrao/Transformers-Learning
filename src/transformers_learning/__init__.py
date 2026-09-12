"""Learning implementation of transformer building blocks."""

# Day 1.2: TL-001 expose the first public tensor-input helpers from the package.
from .tensor_inputs import build_input_ids, describe_input_ids

# Day 2.1: TL-002 expose the first text-to-token-id learning tokenizer.
from .simple_tokenizer import SimpleTokenizer

__all__ = ["SimpleTokenizer", "build_input_ids", "describe_input_ids"]
