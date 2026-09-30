import torch
from torch import Tensor
from llm.gpt.model import GPT
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
    model, idx: torch.Tensor, max_output_tokens, context_length
) -> torch.Tensor:
    for _ in range(max_output_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
