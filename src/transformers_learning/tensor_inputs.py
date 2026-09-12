"""Tensor input utilities introduced on Day 1."""

from __future__ import annotations

import torch


# Day 1.3: TL-001 convert one token-id sequence into the batch-first tensor shape
# expected by transformer models: [batch_size, sequence_length].
def build_input_ids(token_ids: list[int]) -> torch.Tensor:
    if not token_ids:
        raise ValueError("token_ids must contain at least one token id")

    return torch.tensor([token_ids], dtype=torch.long)


# Day 1.4: TL-001 validate and expose the two dimensions that later transformer
# components will depend on.
def describe_input_ids(input_ids: torch.Tensor) -> dict[str, int | str]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be a 2D tensor: [batch_size, sequence_length]")

    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must use torch.long token indices")

    batch_size, sequence_length = input_ids.shape

    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "dtype": str(input_ids.dtype),
    }


# Day 1.5: TL-001 keep a tiny runnable example so the tensor contract can be
# inspected before embeddings and attention are introduced.
def main() -> None:
    token_ids = [101, 2023, 2003, 1037, 3231, 102]
    input_ids = build_input_ids(token_ids)

    print("input_ids:")
    print(input_ids)
    print("shape:", tuple(input_ids.shape))
    print("dtype:", input_ids.dtype)
    print("description:", describe_input_ids(input_ids))


if __name__ == "__main__":
    main()
