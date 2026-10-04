import torch
from gpt.model import GPT
from gpt.tokenizer import Tokenizer

torch.manual_seed(1234)
tokenizer = Tokenizer()

batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "n_positions": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "qkv_bias": False,
}

GPT_CONFIG_124M_MEDIUM = {
    "vocab_size": 50257,
    "n_positions": 1024,
    "n_embd": 1024,
    "n_head": 16,
    "n_layer": 24,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "qkv_bias": False,
}

model = GPT(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

attn_params = sum(
    p.numel()
    for block in model.transformer_blocks
    for p in block.attn.parameters()
    if p.requires_grad
)
print(f"Number of parameters in attention layers: {attn_params:,}")

ffn_params = sum(
    p.numel()
    for block in model.transformer_blocks
    for p in block.ffn.parameters()
    if p.requires_grad
)
print(f"Number of parameters in ffn layers: {ffn_params:,}")


total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")


## GTP-2-MEDIUM
model = GPT(GPT_CONFIG_124M_MEDIUM)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
