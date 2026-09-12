# Transformers Learning Progress

Target: **365 detailed development days**

| Day | Ticket | Engineering problem | Status |
|---:|---|---|---|
| 1 | TL-001 | Represent token IDs with a valid batch-first PyTorch tensor contract | ✅ Complete |
| 2 | TL-002 | Convert raw text into deterministic vocabulary-backed token IDs and reuse the Day 1 tensor contract | ✅ Complete |

## Rules

- One primary engineering problem per day.
- One primary learning commit per day.
- Keep `Day X.Y` markers in source files.
- Do not copy Hugging Face Transformers wholesale.
- Introduce architecture only when the current learning problem requires it.
