import torch
from llm.gpt.model import LayerNorm, MultiHeadAttention, GPT


def test_layernorm_stats():
    m = LayerNorm(emb_dim=16)
    x = torch.randn(2, 8, 16) * 5 + 3  # off mean/var on purpose
    y = m(x)
    assert y.shape == x.shape
    # normalized per-token: mean~0, var~1 before scale/shift (init 1/0)
    assert y.mean(-1).abs().max() < 1e-5
    assert (y.var(-1, unbiased=False) - 1).abs().max() < 1e-4


def test_mha_shape_and_heads():
    m = MultiHeadAttention(
        d_in=12, d_out=12, context_length=8, dropout=0.0, num_heads=3
    )
    m.eval()  # disable dropout for determinism

    x = torch.randn(2, 6, 12)
    assert m(x).shape == (2, 6, 12)


def test_mha_rejects_bad_split():
    try:
        MultiHeadAttention(
            d_in=12, d_out=13, context_length=8, dropout=0.0, num_heads=3
        )
    except AssertionError:
        pass
    else:
        assert False, "should assert d_out % num_heads == 0"


def test_mha_is_causal():
    torch.manual_seed(0)
    m = MultiHeadAttention(
        d_in=8, d_out=8, context_length=6, dropout=0.0, num_heads=2
    )
    m.eval()

    x = torch.randn(1, 6, 8)
    y1 = m(x)
    x2 = x.clone()
    x2[:, -1, :] += 10.0  # perturb only future
    y2 = m(x2)
    # prefix must be unaffected: token i sees only j<=i
    assert torch.allclose(y1[:, :-1, :], y2[:, :-1, :], atol=1e-6)
    assert not torch.allclose(y1[:, -1, :], y2[:, -1, :], atol=1e-6)


def test_gpt_logits_shape():
    cfg = {
        "vocab_size": 20,
        "emb_dim": 12,
        "context_length": 8,
        "n_heads": 3,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    m = GPT(cfg)
    m.eval()

    ids = torch.randint(0, 20, (2, 8))
    logits = m(ids)

    assert logits.shape == (2, 8, 20)  # [B,T,V], row i predicts i+1
