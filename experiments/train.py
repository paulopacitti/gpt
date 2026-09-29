import torch
from llm.gpt.train import loss_dataset
from llm.gpt.data import build_pretraining_dataloader
from llm.gpt.tokenizer import Tokenizer
from llm.gpt.model import GPT
from os import path

DATASET = path.join(path.dirname(__file__), "../data/the-verdict.txt")

with open(DATASET, "r", encoding="utf-8") as file:
    text_data = file.read()

tokenizer = Tokenizer("gpt2")
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

train_loader = build_pretraining_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)
val_loader = build_pretraining_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

model = GPT(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    train_loss = loss_dataset(train_loader, model, device)
    val_loss = loss_dataset(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)
