# gpt

My GPT-2 implementation in PyTorch, based on _Build a Large Language Model (from scratch)_ by Sebastian Raschka.

![ibn-5100](https://i.imgur.com/0RbBI2A.png?1)

## Commands

```bash
# Run the test suite
uv run pytest

# Train or resume from out/pretraining_checkpoint.pth
uv run python train.py

# Generate text from the checkpoint
uv run python generate.py

# Download and generate with the public GPT-2 124M checkpoint
uv run python generate.py --pretrained

# Run the attention and GPT examples
PYTHONPATH=. uv run python examples/attention.py
PYTHONPATH=. uv run python examples/gpt.py
```
