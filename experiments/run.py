import torch
from torch import Tensor
from llm.gpt.model import GPT
from llm.gpt.inference import generate_next_token
from llm.gpt.tokenizer import Tokenizer


torch.manual_seed(123)

def text_to_tokens(text: str, tokenizer: Tokenizer) -> Tensor:
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def tokens_to_text(tokens: Tensor, tokenizer: Tokenizer) -> str:
    flat = tokens.squeeze(dim=0)
    decoded_text = tokenizer.decode(flat.tolist())
    return decoded_text


tokenizer = Tokenizer("gpt2")
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
model = GPT(GPT_CONFIG_124M)
model.eval()
start_sequence = "Every effort moves you"

sequence = generate_next_token(
    model,
    sequence=text_to_tokens(start_sequence, tokenizer),
    max_output_tokens=10,
    context_length=GPT_CONFIG_124M["context_length"],
)

print("Output:", sequence)
print("Output length:", len(sequence[0]))
print(tokens_to_text(sequence, tokenizer))
