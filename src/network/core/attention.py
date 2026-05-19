import math

import torch

from src.network.core.utils import (
    _build_transformer_ffn,
    _init_linear,
    _merge_heads,
    _scaled_dot_product_attention,
    _split_heads,
)


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
        mask : bool tensor; True = keep, False = mask out. Must be broadcastable
               to the attention scores shape (..., num_heads, Lq, Lk). As a
               convenience, a mask with exactly one fewer dim than the scores
               (i.e. no head dim) is auto-unsqueezed at position -3 and thus
               treated as head-agnostic. Any other rank must already be
               broadcast-compatible with (..., num_heads, Lq, Lk).
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
        mask : bool tensor; True = keep, False = mask out. Must be broadcastable
               to the attention scores shape (..., num_heads, L, L). As a
               convenience, a mask with exactly one fewer dim than the scores
               (i.e. no head dim) is auto-unsqueezed at position -3 and thus
               treated as head-agnostic. Any other rank must already be
               broadcast-compatible with (..., num_heads, L, L).
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
    """Pre-LN transformer-style cross-attention block with residual attention and FFN sublayers."""

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
        q = q + self.attention(self.ln_attn(q), kv, mask=mask)
        return q + self.ffn(self.ln_ffn(q))


class ResidualSelfAttentionBlock(torch.nn.Module):
    """Pre-LN transformer-style self-attention block with fused-QKV attention and FFN sublayers."""

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
        tokens = tokens + self.attention(self.ln_attn(tokens), mask=mask)
        return tokens + self.ffn(self.ln_ffn(tokens))


__all__ = [
    "CrossAttention",
    "SelfAttention",
    "ResidualCrossAttentionBlock",
    "ResidualSelfAttentionBlock",
]
