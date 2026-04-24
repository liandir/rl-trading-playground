import math

import torch


def _split_heads(t: torch.Tensor, num_heads: int, d_head: int) -> torch.Tensor:
    """(..., L, inner_dim) -> (..., num_heads, L, d_head)"""
    return t.unflatten(-1, (num_heads, d_head)).transpose(-3, -2)


def _merge_heads(t: torch.Tensor) -> torch.Tensor:
    """(..., num_heads, L, d_head) -> (..., L, inner_dim)"""
    return t.transpose(-3, -2).flatten(-2)


def _scaled_dot_product_attention(
    q_h: torch.Tensor,
    k_h: torch.Tensor,
    v_h: torch.Tensor,
    *,
    scale: float,
    mask: torch.Tensor | None,
    attn_dropout: float,
    training: bool,
) -> torch.Tensor:
    scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale  # (..., H, Lq, Lk)
    if mask is not None:
        if mask.dim() == scores.dim() - 1:
            mask = mask.unsqueeze(-3)
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    if attn_dropout > 0.0 and training:
        attn = torch.nn.functional.dropout(attn, p=attn_dropout)
    return torch.matmul(attn, v_h)  # (..., H, Lq, Lk) @ (..., H, Lk, D)


def _init_linear(linear: torch.nn.Linear, *, bias: bool) -> None:
    torch.nn.init.xavier_uniform_(linear.weight)
    if bias:
        torch.nn.init.zeros_(linear.bias)


def _build_transformer_ffn(
    d_model: int,
    ff_hidden_dim: int,
    *,
    ff_dropout: float,
    bias: bool,
) -> torch.nn.Sequential:
    ffn = torch.nn.Sequential(
        torch.nn.Linear(d_model, ff_hidden_dim, bias=bias),
        torch.nn.GELU(),
        torch.nn.Dropout(ff_dropout),
        torch.nn.Linear(ff_hidden_dim, d_model, bias=bias),
        torch.nn.Dropout(ff_dropout),
    )
    for module in ffn:
        if isinstance(module, torch.nn.Linear):
            _init_linear(module, bias=bias)
    return ffn


class CrossAttention(torch.nn.Module):
    """
    Multi-head scaled dot-product cross-attention.

    Implements standard attention:
        out = softmax(Q K^T / sqrt(d_head)) V

    Accepts queries and keys/values with arbitrary leading batch dimensions:
        q  : (..., Lq, d_model)
        kv : (..., Lk, d_kv)
        ->  (..., Lq, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        d_kv: int | None = None,
        d_head: int | None = None,
        attn_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}.")

        d_kv = d_model if d_kv is None else d_kv
        if d_head is None:
            if d_model % num_heads != 0:
                raise ValueError(
                    f"d_model ({d_model}) must be divisible by num_heads ({num_heads}) "
                    "when d_head is not specified."
                )
            d_head = d_model // num_heads

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_kv = int(d_kv)
        self.d_head = int(d_head)
        self.inner_dim = self.num_heads * self.d_head
        self.attn_dropout = float(attn_dropout)
        self._scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = torch.nn.Linear(self.d_model, self.inner_dim, bias=bias)
        self.k_proj = torch.nn.Linear(self.d_kv, self.inner_dim, bias=bias)
        self.v_proj = torch.nn.Linear(self.d_kv, self.inner_dim, bias=bias)
        self.out_proj = torch.nn.Linear(self.inner_dim, self.d_model, bias=bias)

        _init_linear(self.q_proj, bias=bias)
        _init_linear(self.k_proj, bias=bias)
        _init_linear(self.v_proj, bias=bias)
        _init_linear(self.out_proj, bias=bias)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        q    : (..., Lq, d_model)
        kv   : (..., Lk, d_kv)
        mask : broadcastable to (..., Lq, Lk); True = keep, False = mask out.
               If mask has one fewer dim than the attention scores, it is
               treated as head-agnostic and broadcast across heads.
        returns (..., Lq, d_model)
        """
        q_h = _split_heads(self.q_proj(q), self.num_heads, self.d_head)    # (..., H, Lq, D)
        k_h = _split_heads(self.k_proj(kv), self.num_heads, self.d_head)   # (..., H, Lk, D)
        v_h = _split_heads(self.v_proj(kv), self.num_heads, self.d_head)   # (..., H, Lk, D)

        out = _scaled_dot_product_attention(
            q_h,
            k_h,
            v_h,
            scale=self._scale,
            mask=mask,
            attn_dropout=self.attn_dropout,
            training=self.training,
        )
        out = _merge_heads(out)  # (..., Lq, inner_dim)
        return self.out_proj(out)


class SelfAttention(torch.nn.Module):
    """
    Multi-head scaled dot-product self-attention with fused QKV projection.

    Implements standard attention:
        out = softmax(Q K^T / sqrt(d_head)) V

    Accepts inputs with arbitrary leading batch dimensions:
        x : (..., L, d_model)
        ->  (..., L, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        d_head: int | None = None,
        attn_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}.")

        if d_head is None:
            if d_model % num_heads != 0:
                raise ValueError(
                    f"d_model ({d_model}) must be divisible by num_heads ({num_heads}) "
                    "when d_head is not specified."
                )
            d_head = d_model // num_heads

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_head = int(d_head)
        self.inner_dim = self.num_heads * self.d_head
        self.attn_dropout = float(attn_dropout)
        self._scale = 1.0 / math.sqrt(self.d_head)

        self.qkv_proj = torch.nn.Linear(self.d_model, 3 * self.inner_dim, bias=bias)
        self.out_proj = torch.nn.Linear(self.inner_dim, self.d_model, bias=bias)

        _init_linear(self.qkv_proj, bias=bias)
        _init_linear(self.out_proj, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x    : (..., L, d_model)
        mask : broadcastable to (..., L, L); True = keep, False = mask out.
               If mask has one fewer dim than the attention scores, it is
               treated as head-agnostic and broadcast across heads.
        returns (..., L, d_model)
        """
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q_h = _split_heads(q, self.num_heads, self.d_head)  # (..., H, L, D)
        k_h = _split_heads(k, self.num_heads, self.d_head)  # (..., H, L, D)
        v_h = _split_heads(v, self.num_heads, self.d_head)  # (..., H, L, D)

        out = _scaled_dot_product_attention(
            q_h,
            k_h,
            v_h,
            scale=self._scale,
            mask=mask,
            attn_dropout=self.attn_dropout,
            training=self.training,
        )
        out = _merge_heads(out)  # (..., L, inner_dim)
        return self.out_proj(out)


class ResidualCrossAttentionBlock(torch.nn.Module):
    """Transformer-style cross-attention block with residual attention and FFN sublayers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        *,
        d_kv: int | None = None,
        ff_hidden_dim: int | None = None,
        use_layer_norm: bool = True,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        d_model = int(d_model)
        ff_hidden_dim = 4 * d_model if ff_hidden_dim is None else int(ff_hidden_dim)
        if ff_hidden_dim < 1:
            raise ValueError(f"ff_hidden_dim must be >= 1, got {ff_hidden_dim}.")

        self.attention = CrossAttention(
            d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            attn_dropout=attn_dropout,
            bias=bias,
        )
        self.ln_attn = torch.nn.LayerNorm(d_model) if use_layer_norm else torch.nn.Identity()
        self.ln_ffn = torch.nn.LayerNorm(d_model) if use_layer_norm else torch.nn.Identity()
        self.ffn = _build_transformer_ffn(
            d_model,
            ff_hidden_dim,
            ff_dropout=ff_dropout,
            bias=bias,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.ln_attn(q + self.attention(q, kv, mask=mask))
        return self.ln_ffn(q + self.ffn(q))


class ResidualSelfAttentionBlock(torch.nn.Module):
    """Transformer-style self-attention block with fused-QKV attention and FFN sublayers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        *,
        ff_hidden_dim: int | None = None,
        use_layer_norm: bool = True,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        d_model = int(d_model)
        ff_hidden_dim = 4 * d_model if ff_hidden_dim is None else int(ff_hidden_dim)
        if ff_hidden_dim < 1:
            raise ValueError(f"ff_hidden_dim must be >= 1, got {ff_hidden_dim}.")

        self.attention = SelfAttention(
            d_model,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            bias=bias,
        )
        self.ln_attn = torch.nn.LayerNorm(d_model) if use_layer_norm else torch.nn.Identity()
        self.ln_ffn = torch.nn.LayerNorm(d_model) if use_layer_norm else torch.nn.Identity()
        self.ffn = _build_transformer_ffn(
            d_model,
            ff_hidden_dim,
            ff_dropout=ff_dropout,
            bias=bias,
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.ln_attn(tokens + self.attention(tokens, mask=mask))
        return self.ln_ffn(tokens + self.ffn(tokens))


__all__ = [
    "CrossAttention",
    "ResidualCrossAttentionBlock",
    "ResidualSelfAttentionBlock",
    "SelfAttention",
]
