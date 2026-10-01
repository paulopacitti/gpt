import torch
from torch import Tensor
from llm.gpt.tokenizer import Tokenizer


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
    """Autoregressively extend `sequence` by up to `max_output_tokens` ids.

    Each step crops to `sequence[:, -context_length:]`, runs `model`
    (`[B,T] -> [B,T,V]`), and decodes the last-position logits `[B,V]`.
    With `top_k`, logits below the k-th largest are set to -inf.
    With `temperature > 0`, samples `multinomial(softmax(logits/T))`;
    otherwise greedy `argmax`. Breaks (discarding eos) when the next
    id equals `eos_id`. Caller sets `model.eval()` and places
    `sequence` on the model's device. Returns `[B, T+n]`.
    """
    for _ in range(max_output_tokens):
        idx_cond = sequence[:, -context_length:]
        # generate next token
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
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
