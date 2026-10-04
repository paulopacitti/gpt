import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Any

from matplotlib.ticker import MaxNLocator
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from gpt.data import build_pretraining_dataloader
from gpt.model import GPT
from gpt.tokenizer import Tokenizer
from generate import generate_next_token, text_to_tokens, tokens_to_text

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT.joinpath("data", "the-verdict.txt")
CHECKPOINT_PATH = ROOT.joinpath("out", "pretraining_checkpoint.pth")

GPT_CONFIG_124M: dict[str, Any] = {
    "vocab_size": 50257,
    "n_positions": 256,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "qkv_bias": False,
}


def loss_batch(inputs: Tensor, targets: Tensor, model, device) -> Tensor:
    inputs = inputs.to(device)
    targets = targets.to(device)

    logits: Tensor = model(inputs)
    loss = cross_entropy(logits.flatten(0, 1), targets.flatten())
    return loss


def loss_dataset(data_loader, model, device, num_batches=None) -> float:
    """Average cross-entropy over up to `num_batches` batches."""
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0
    for i, (inputs, targets) in enumerate(data_loader):
        if i < num_batches:
            loss = loss_batch(inputs, targets, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model, train_dataloader, val_dataloader, device, eval_iter
) -> tuple[float, float]:
    """Estimate mean train and validation loss over up to `eval_iter` batches."""
    model.eval()
    with torch.no_grad():
        train_loss = loss_dataset(
            train_dataloader, model, device, num_batches=eval_iter
        )
        val_loss = loss_dataset(val_dataloader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def log_text_sample(model: GPT, tokenizer, device, sequence) -> None:
    model.eval()
    with torch.no_grad():
        context_length = model.pos_emb.weight.shape[0]
        encoded = text_to_tokens(sequence, tokenizer).to(device)
        token_ids = generate_next_token(
            model=model,
            sequence=encoded,
            max_output_tokens=50,
            context_length=context_length,
        )
        decoded_text = tokens_to_text(token_ids.cpu(), tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    tokens_seen: int,
    path: str,
    config: dict[str, Any],
) -> None:
    """Persist model and optimizer state for resuming pretraining."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "tokens_seen": tokens_seen,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> dict:
    """Restore model and optimizer state."""
    checkpoint = torch.load(path, map_location=device)
    assert "model_state_dict" in checkpoint and "optimizer_state_dict" in checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
    return checkpoint


def plot_loss(epochs_seen, tokens_seen, train_acc_loss, val_acc_loss) -> None:
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_acc_loss, label="training loss")
    ax1.plot(epochs_seen, val_acc_loss, label="validation loss")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_acc_loss, alpha=0)
    ax2.set_xlabel("tokens seen")
    fig.tight_layout()
    plt.show()


def train(
    model,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    sequence,
    tokenizer,
):
    """Run next-token training with periodic evaluation and sampling."""
    train_acc_loss, val_acc_loss, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            loss = loss_batch(inputs, targets, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += inputs.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader, device, eval_iter
                )
                train_acc_loss.append(train_loss)
                val_acc_loss.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss: {train_loss:.3f} "
                    f"Validation loss: {val_loss:.3f}"
                )

    log_text_sample(model, tokenizer, device=device, sequence=sequence)
    return train_acc_loss, val_acc_loss, track_tokens_seen


def main() -> None:
    torch.manual_seed(123)

    text_data = DATASET_PATH.read_text(encoding="utf-8")
    tokenizer = Tokenizer()
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = build_pretraining_dataloader(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["n_positions"],
        stride=GPT_CONFIG_124M["n_positions"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = build_pretraining_dataloader(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["n_positions"],
        stride=GPT_CONFIG_124M["n_positions"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    model = GPT(GPT_CONFIG_124M)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    start_epoch = 0
    if CHECKPOINT_PATH.exists():
        checkpoint = load_checkpoint(model, optimizer, str(CHECKPOINT_PATH), device)
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"Resumed from {CHECKPOINT_PATH} (epoch {checkpoint.get('epoch')})")

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        sequence="Every effort moves you",
        tokenizer=tokenizer,
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model,
        optimizer,
        epoch=start_epoch + num_epochs - 1,
        tokens_seen=tokens_seen[-1] if len(tokens_seen) > 0 else 0,
        path=str(CHECKPOINT_PATH),
        config=GPT_CONFIG_124M,
    )
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_loss(epochs_tensor, tokens_seen, train_losses, val_losses)


if __name__ == "__main__":
    main()
