import json
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch import Tensor, nn

from gpt.model import GPT

ROOT = Path(__file__).resolve().parent
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
WEIGHTS_URL = (
    f"https://huggingface.co/openai-community/gpt2/resolve/{REVISION}/model.safetensors"
)
WEIGHTS_PATH = ROOT.joinpath("out", "pretrained", "gpt2", "model.safetensors")

SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}

GPT2_CONFIG = {
    "vocab_size": 50257,
    "n_positions": 1024,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "embd_pdrop": 0.0,
    "attn_pdrop": 0.0,
    "resid_pdrop": 0.0,
    "qkv_bias": True,
}


def download_weights(path: Path) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading GPT-2 weights to {path}")
    urlretrieve(WEIGHTS_URL, path)


def load_safetensors(path: Path) -> dict[str, Tensor]:
    """Read tensors from a safetensors file into CPU PyTorch tensors.

    Safetensors files start with an 8-byte little-endian header length, followed
    by a JSON header and then the raw tensor data. Each header entry gives a
    tensor's dtype, shape, and byte offsets within that data section.

    This function reads and parses the header, then processes each tensor entry:
    it skips file metadata, seeks to the tensor's byte range, wraps those bytes
    in a buffer, and creates a PyTorch tensor with the recorded dtype and shape.
    """
    with path.open("rb") as file:
        length_bytes = file.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Truncated safetensors header")
        header_size = int.from_bytes(length_bytes, byteorder="little")
        header = json.loads(file.read(header_size))
        data_start = file.tell()
        tensors = {}

        for name, metadata in header.items():
            if name == "__metadata__":
                continue

            start, end = metadata["data_offsets"]
            file.seek(data_start + start)
            raw_tensor = bytearray(file.read(end - start))
            dtype = SAFETENSORS_DTYPES[metadata["dtype"]]
            tensors[name] = torch.frombuffer(raw_tensor, dtype=dtype).reshape(
                metadata["shape"]
            )

    return tensors


def assign(destination: Tensor, source: Tensor) -> None:
    if destination.shape != source.shape:
        raise ValueError(
            f"Weight shape mismatch: expected {destination.shape}, got {source.shape}"
        )
    destination.copy_(source.to(dtype=destination.dtype))


def assign_weights_into_gpt(model: GPT, weights: dict[str, Tensor]) -> None:
    with torch.no_grad():
        assign(model.tok_emb.weight, weights["wte.weight"])
        assign(model.pos_emb.weight, weights["wpe.weight"])

        for index, block in enumerate(model.transformer_blocks):
            prefix = f"h.{index}."
            qkv_weight = weights[prefix + "attn.c_attn.weight"].T
            q_weights, k_weights, v_weights = qkv_weight.chunk(3, dim=0)
            q_biases, k_biases, v_biases = weights[prefix + "attn.c_attn.bias"].chunk(3)

            assign(block.attn.W_query.weight, q_weights)
            assign(block.attn.W_key.weight, k_weights)
            assign(block.attn.W_value.weight, v_weights)

            assign(block.attn.W_query.bias, q_biases)
            assign(block.attn.W_key.bias, k_biases)
            assign(block.attn.W_value.bias, v_biases)

            assign(
                block.attn.out_proj.weight,
                weights[prefix + "attn.c_proj.weight"].T,
            )
            assign(block.attn.out_proj.bias, weights[prefix + "attn.c_proj.bias"])

            for norm, name in ((block.norm1, "ln_1"), (block.norm2, "ln_2")):
                assign(norm.scale, weights[prefix + name + ".weight"])
                assign(norm.shift, weights[prefix + name + ".bias"])

            assign(block.ffn.layers[0].weight, weights[prefix + "mlp.c_fc.weight"].T)
            assign(block.ffn.layers[0].bias, weights[prefix + "mlp.c_fc.bias"])
            assign(
                block.ffn.layers[2].weight,
                weights[prefix + "mlp.c_proj.weight"].T,
            )
            assign(block.ffn.layers[2].bias, weights[prefix + "mlp.c_proj.bias"])

        assign(model.final_norm.scale, weights["ln_f.weight"])
        assign(model.final_norm.shift, weights["ln_f.bias"])



def load_gpt2_124m(
    device: torch.device | str = "cpu",
    weights_path: Path = WEIGHTS_PATH,
) -> GPT:
    """Download and load OpenAI's public 124M GPT-2 weights."""
    download_weights(weights_path)
    weights = load_safetensors(weights_path)

    model = GPT(GPT2_CONFIG)
    # GPT-2 shares its token embedding and output projection weights.
    model.out_head.weight = model.tok_emb.weight
    # GPT-2 uses the tanh approximation known as gelu_new.
    for layer in model.modules():
        if isinstance(layer, nn.GELU):
            setattr(layer, "approximate", "tanh")

    assign_weights_into_gpt(model, weights)
    del weights
    return model.to(device).eval()
