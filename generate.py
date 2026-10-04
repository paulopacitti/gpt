import argparse
from pathlib import Path
import torch
from torch import Tensor

from gpt.model import GPT
from gpt.tokenizer import Tokenizer
from pretrained import load_gpt2_124m

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT.joinpath("out", "pretraining_checkpoint.pth")

def text_to_tokens(text: str, tokenizer: Tokenizer) -> Tensor:
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def tokens_to_text(tokens: Tensor, tokenizer: Tokenizer) -> str:
    flat = tokens.squeeze(dim=0)
    decoded_text = tokenizer.decode(flat.tolist())
    return decoded_text


def generate_next_token(
    model,
    sequence: torch.Tensor,
    max_output_tokens,
    context_length,
    temperature=0.0,
    top_k=None,
    eos_id=None,
) -> torch.Tensor:
    """Autoregressively extend `sequence` by up to `max_output_tokens` ids."""
    for _ in range(max_output_tokens):
        idx_cond = sequence[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )
        if temperature > 0.0:
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        sequence = torch.cat((sequence, idx_next), dim=1)

    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with GPT-2.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Download and use the public GPT-2 124M checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    tokenizer = Tokenizer()
    if args.pretrained:
        model = load_gpt2_124m(device)
        context_length = model.pos_emb.weight.shape[0]
    else:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        config = checkpoint["config"]
        model = GPT(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        context_length = config["n_positions"]

    start_sequence = "Every effort moves you"
    sequence = generate_next_token(
        model,
        sequence=text_to_tokens(start_sequence, tokenizer).to(device),
        max_output_tokens=50,
        context_length=context_length,
        temperature=1.5,
        top_k=50
    )

    print(tokens_to_text(sequence.cpu(), tokenizer))


if __name__ == "__main__":
    main()
