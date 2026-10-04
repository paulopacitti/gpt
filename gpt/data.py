"""Dataset and dataloader utilities for next-token pretraining."""

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from .tokenizer import Tokenizer


class PretrainingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Create fixed-length input and target windows from text.

    Each target window is shifted one token forward from its input window.
    `stride` sets how far the start moves between windows.
    """

    def __init__(
        self,
        text: str,
        tokenizer: Tokenizer,
        max_length: int,
        stride: int,
    ):
        self.tokenizer = tokenizer
        self.input_ids: list[Tensor] = []
        self.target_ids: list[Tensor] = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.input_ids[index], self.target_ids[index]


def build_pretraining_dataloader(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build batches of next-token examples from raw text.

    Creates a default GPT-2 tokenizer and uses the given settings to batch
    the dataset windows.
    """
    tokenizer = Tokenizer()
    dataset = PretrainingDataset(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    )
