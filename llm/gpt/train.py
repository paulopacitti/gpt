from llm.gpt.model import GPT
from llm.gpt.inference import generate_next_token, text_to_tokens, tokens_to_text
from typing import Tuple
import torch
from torch.nn.functional import cross_entropy
from torch import Tensor
from torch.utils.data import DataLoader


# Calculates the loss of a batch (e.g. [B, S, E])
def loss_batch(inputs: Tensor, targets: Tensor, model, device) -> Tensor:
    inputs = inputs.to(device)
    targets = targets.to(device)

    logits: Tensor = model(inputs)
    loss = cross_entropy(logits.flatten(0, 1), targets.flatten())
    return loss


def loss_dataset(data_loader, model, device, num_batches=None) -> float:
    """Average cross-entropy over up to `num_batches` batches.

    Uses all batches when `num_batches` is None, else clamps to
    `len(data_loader)`. Returns NaN for an empty loader. Caller
    handles eval/no-grad; per-batch loss comes from `loss_batch`.
    """
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
) -> Tuple[float, float]:
    """Estimate mean train/val loss over up to `eval_iter` batches.

    Switches model to eval mode (disables dropout) under `no_grad`,
    averages `loss_dataset` on each loader, then restores train mode.
    Returns (train_loss, val_loss); NaN if a loader is empty.
    """
    model.eval()
    with torch.no_grad():
        train_loss = loss_dataset(
            train_dataloader, model, device, num_batches=eval_iter
        )
        val_loss = loss_dataset(val_dataloader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def log_text_sample(model: GPT, tokenizer, device, sequence):
    model.eval()
    with torch.no_grad():
        context_length = model.pos_emb.weight.shape[0]
        encoded = text_to_tokens(sequence, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate_next_token(
                model=model,
                sequence=encoded,
                max_output_tokens=50,
                context_length=context_length,
            )
        decoded_text = tokens_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
        model.train()


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    tokens_seen: int,
    path: str,
) -> None:
    """Persist model + optimizer state for resuming pretraining."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "tokens_seen": tokens_seen,
        },
        path,
    )


def load_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> dict:
    """Restore model + optimizer state; leaves model in train mode.

    Optimizer must be constructed first with the same hparams.
    Returns the raw checkpoint dict (epoch, tokens_seen).
    """
    ckpt = torch.load(path, map_location=device)
    assert "model_state_dict" in ckpt and "optimizer_state_dict" in ckpt
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    model.train()
    return ckpt


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
    """Run NTP training loop with periodic eval and final sampling.

    Iterates `num_epochs` over `train_dataloader`, doing
    zero_grad/backward/step per batch while counting `tokens_seen`.
    Every `eval_freq` steps, records mean losses via `evaluate_model`
    (up to `eval_iter` batches each). Ends with `log_text_sample`
    from `sequence`. Returns (train_losses, val_losses, tokens_seen).
    """
    train_acc_loss, val_acc_loss, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for (
            inputs,
            targets,
        ) in train_dataloader:
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
