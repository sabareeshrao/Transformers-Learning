# Transformers Learning

Hands-on reconstruction of core ideas and engineering patterns from Hugging Face Transformers.

Reference repository: `huggingface/transformers`

This repository does not copy the finished library wholesale. It rebuilds important transformer concepts in a logical learning order through small, working commits.

## Development Method

Each day represents one primary engineering problem and one meaningful learning commit.

Every day includes:

- engineering group discussion
- design decision
- exact files and code
- runnable verification
- tests
- Day X.Y source markers
- Git commit and push verification

## Source History Markers

Code introduced during the journey is marked with comments such as:

```python
# Day 1.3: TL-001 convert token ids into a batch-first tensor
# Day 2.3: TL-002 introduce deterministic lowercase whitespace tokenization
```

These markers remain in the source so later files show how the implementation evolved over time.

## Current Progress

- Target: 365 detailed days
- Completed: Day 2 / 365
- Latest ticket: `TL-002`
- Latest topic: raw text, tokenization, vocabulary lookup, unknown-token fallback, and reuse of the Day 1 tensor contract

The earlier `TEST Verify Transformers-Learning push access` commit is only a connection check and does not count as a learning day.
