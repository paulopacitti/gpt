import json
import struct

import torch

from pretrained import load_safetensors


def test_load_safetensors(tmp_path):
    tensor_data = struct.pack("<2f", 1.5, -2.0)
    header = {
        "weight": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [0, len(tensor_data)],
        },
        "__metadata__": {"format": "pt"},
    }
    header_data = json.dumps(header).encode()
    header_data += b" " * (-len(header_data) % 8)
    path = tmp_path / "weights.safetensors"
    path.write_bytes(struct.pack("<Q", len(header_data)) + header_data + tensor_data)

    weights = load_safetensors(path)

    assert torch.equal(weights["weight"], torch.tensor([1.5, -2.0]))
