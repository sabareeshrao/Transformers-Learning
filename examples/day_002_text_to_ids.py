"""Run the Day 2 raw-text to model-input demonstration."""

from transformers_learning import SimpleTokenizer


# Day 2.7: TL-002 demonstrate raw text flowing through tokenization, vocabulary
# lookup, unknown-token fallback, and the Day 1 tensor contract.
def main() -> None:
    vocab = {
        "[UNK]": 0,
        "transformers": 1,
        "turn": 2,
        "text": 3,
        "into": 4,
        "tokens": 5,
    }
    tokenizer = SimpleTokenizer(vocab)
    text = "Transformers turn text into attention"

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.encode_to_tensor(text)

    print("text:", text)
    print("tokens:", tokens)
    print("token_ids:", token_ids)
    print("input_ids:")
    print(input_ids)
    print("shape:", tuple(input_ids.shape))
    print("dtype:", input_ids.dtype)


if __name__ == "__main__":
    main()
