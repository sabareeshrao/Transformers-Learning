"""Learning implementation of transformer building blocks."""

# Day 1.2: TL-001 expose the first public tensor-input helpers from the package.
from .tensor_inputs import build_input_ids, describe_input_ids

__all__ = ["build_input_ids", "describe_input_ids"]
