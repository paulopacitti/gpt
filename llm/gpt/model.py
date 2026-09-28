from torch import nn, Tensor
import torch


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        z = attention_weights @ values  # context vector
        return z


class CausalAttention(nn.Module):
    mask: torch.Tensor

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        z = attention_weights @ values
        return z


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention.

    Splits emb_dim into num_heads x head_dim, runs attention
    independently per head, then merges. Allows each head to
    learn a different position-to-position routing.

    Shapes: B=batch, T=tokens, D=emb_dim, H=heads, Dh=head_dim.
    """
    mask: torch.Tensor 

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """Init projections, output mix, causal mask.

        d_in: input dim (emb_dim). d_out: output dim (emb_dim).
        context_length: max T, sizes the causal mask.
        dropout: dropout on attn weights. num_heads: H, must divide d_out.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out  must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # Upper-triangular 1s: mask[j>k]=1 blocks future, [L,L]
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x) -> Tensor:
        """x: [B,T,D] -> z: [B,T,D], each pos mixes only past."""
        b, num_tokens, d_in = x.shape
        # [B,T,D] each: raw Q/K/V before head split
        queries: Tensor = self.W_query(x)
        keys: Tensor = self.W_key(x)
        values: Tensor = self.W_value(x)

        # Multi head attention
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # Scores per head: [B,H,T,T], score[i,j]=q_i dot k_j
        attn_scores = queries @ keys.transpose(2, 3)
        # Slice mask to actual T, fill future with -inf so softmax->0
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights: Tensor = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights: Tensor = self.dropout(attn_weights)
        z = (attn_weights @ values).transpose(1, 2)

        # Glue heads: [B,T,H,Dh] -> [B,T,D]
        z = z.reshape(b, num_tokens, self.d_out)
        # Learned blend across heads
        z: Tensor = self.out_proj(z)
        return z


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ffn = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

        return x


class GPT(nn.Module):
    context_length: int
    token_embedding_layer: nn.Embedding
    positional_embedding_layer: nn.Embedding

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x) -> Tensor:
        batch_size, seq_len = x.shape
        tok_embeds: Tensor = self.tok_emb(x)
        pos_embeds: Tensor= self.pos_emb(torch.arange(seq_len, device=x.device))

        x = tok_embeds + pos_embeds
        x: Tensor = self.drop_emb(x)
        x: Tensor = self.transformer_blocks(x)
        x: Tensor = self.final_norm(x)
        logits: Tensor = self.out_head(x)
        return logits
