# AGENTS.md

GPT implementation in PyTorch. Based on "Build a Large Language Model (from scratch)" by Sebastian Raschka.

## Project Structure

```
gpt/                     # GPT model package
├── __init__.py
├── model.py             # GPT and educational attention implementations
├── data.py              # Dataset and dataloader utilities
└── tokenizer.py         # Tokenization utilities

train.py                 # Training loop and entry point
generate.py              # Text generation entry point
pretrained.py            # Public GPT-2 weight download and loading
examples/                # Standalone educational examples

data/                    # Data files (outside library)
```

## Build/Run Commands

```bash
# Install dependencies
uv sync

# Train or resume from the checkpoint in out/
uv run python train.py

# Generate text from the checkpoint in out/
uv run python generate.py

# Download and generate with the public GPT-2 124M checkpoint
uv run python generate.py --pretrained

# Run an educational example
PYTHONPATH=. uv run python examples/attention.py

# Type checking (linter)
uv run ty check

# Run Python REPL
uv run python
```

## Code Style

- Use assertions for internal invariants
- Add type hints to function signatures
```python
- Return types are encouraged but optional for obvious returns
- Use relative imports within the package (`from .tokenizer import Tokenizer`)

## Notes

- Training configuration is saved with checkpoints for sampling
- Scripts derive data and checkpoint paths from their own file locations
- Tokenizer uses `tiktoken` with `gpt2` encoding by default; implement from scratch later
