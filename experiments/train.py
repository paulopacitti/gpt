import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from llm.gpt.train import train, save_checkpoint, load_checkpoint
from llm.gpt.data import build_pretraining_dataloader
from llm.gpt.tokenizer import Tokenizer
from llm.gpt.model import GPT
from os import path

def plot_loss(epochs_seen, tokens_seen, train_acc_loss, val_acc_loss):
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen, train_acc_loss, label="training loss")
    ax1.plot(epochs_seen, val_acc_loss, label="validation loss")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax1.legend("upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_acc_loss, alpha=0)
    ax2.set_xlabel("tokens seen")
    fig.tight_layout()
    plt.show()

torch.manual_seed(123)
DATASET = path.join(path.dirname(__file__), "../data/the-verdict.txt")
CHECKPOINT_PATH = path.join(path.dirname(__file__), "../checkpoints/pretraining_checkpoint.pth")

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
device = torch.device("mps" if torch.mps.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(
     model.parameters(),
    lr=0.0004, weight_decay=0.1
)

start_epoch = 0
if path.exists(CHECKPOINT_PATH):
    ckpt = load_checkpoint(model, optimizer, CHECKPOINT_PATH, device)
    start_epoch = ckpt.get("epoch", -1) + 1
    print(f"Resumed from {CHECKPOINT_PATH} (epoch {ckpt.get('epoch')})")

num_epochs = 10
train_losses, val_losses, tokens_seen = train(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    sequence="Every effort moves you", tokenizer=tokenizer
)

save_checkpoint(
    model, optimizer,
    epoch=start_epoch + num_epochs - 1,
    tokens_seen=tokens_seen[-1] if len(tokens_seen) > 0 else 0,
    path=CHECKPOINT_PATH,
)
print(f"Saved checkpoint to {CHECKPOINT_PATH}")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_loss(epochs_tensor, tokens_seen, train_losses, val_losses)

