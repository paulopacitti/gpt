from torch.nn.functional import cross_entropy
from torch import Tensor

# Calculates the loss of a batch (e.g. [B, S, E])
def loss_batch(inputs: Tensor, targets: Tensor, model, device) -> Tensor:
    inputs = inputs.to(device)
    targets = targets.to(device)

    logits: Tensor = model(inputs)
    loss = cross_entropy(logits.flatten(0, 1), targets.flatten())
    return loss

# Calculates the accumulated loss for the whole dataset loaded in the data_loader
def loss_dataset(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches == min(num_batches, len(data_loader))


    total_loss = 0
    for i, (inputs, targets) in enumerate(data_loader):
        if i < num_batches:
            loss = loss_batch(inputs, targets, model, device)
            total_loss += loss.item()

        else:
            break
    return total_loss / num_batches
