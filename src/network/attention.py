import math

import torch


class CrossAttention(torch.nn.Module):
    """
    Single-head scaled dot-product cross-attention.

        out = softmax(Q K^T / sqrt(d_head)) V

    Accepts queries and keys/values with arbitrary leading batch dimensions:
        q  : (..., Lq, d_model)
        kv : (..., Lk, d_kv)
        →  (..., Lq, d_model)

    When ``q is kv`` this reduces to self-attention.

    Parameters
    ----------
    d_model   : feature dim of queries and of the output.
    d_kv      : feature dim of keys/values. Defaults to ``d_model``.
    d_head    : inner dim used by Q/K/V projections. Defaults to ``d_model``.
    attn_dropout : dropout probability applied to the softmax attention weights.
    bias      : whether Q/K/V/out projections include biases.
    """

    def __init__(
        self,
        d_model: int,
        d_kv: int | None = None,
        d_head: int | None = None,
        attn_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        d_kv = d_model if d_kv is None else d_kv
        d_head = d_model if d_head is None else d_head

        self.d_model = d_model
        self.d_kv = d_kv
        self.d_head = d_head
        self.attn_dropout = float(attn_dropout)
        self._scale = 1.0 / math.sqrt(d_head)

        self.q_proj = torch.nn.Linear(d_model, d_head, bias=bias)
        self.k_proj = torch.nn.Linear(d_kv, d_head, bias=bias)
        self.v_proj = torch.nn.Linear(d_kv, d_head, bias=bias)
        self.out_proj = torch.nn.Linear(d_head, d_model, bias=bias)

        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        torch.nn.init.xavier_uniform_(self.v_proj.weight)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        if bias:
            torch.nn.init.zeros_(self.q_proj.bias)
            torch.nn.init.zeros_(self.k_proj.bias)
            torch.nn.init.zeros_(self.v_proj.bias)
            torch.nn.init.zeros_(self.out_proj.bias)

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
        returns (..., Lq, d_model)
        """
        q_h = self.q_proj(q)        # (..., Lq, d_head)
        k_h = self.k_proj(kv)       # (..., Lk, d_head)
        v_h = self.v_proj(kv)       # (..., Lk, d_head)

        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * self._scale  # (..., Lq, Lk)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        if self.attn_dropout > 0.0 and self.training:
            attn = torch.nn.functional.dropout(attn, p=self.attn_dropout)
        out = torch.matmul(attn, v_h)                                    # (..., Lq, d_head)
        return self.out_proj(out)                                        # (..., Lq, d_model)


class MultiHeadCrossAttention(torch.nn.Module):
    """
    Multi-head scaled dot-product cross-attention.

    Implements the standard transformer multi-head attention:
        inner_dim = num_heads * d_head
        Q/K/V are projected to ``inner_dim`` then split into ``num_heads``
        heads of dim ``d_head``; attention runs per-head in parallel; heads
        are concatenated and projected back to ``d_model``.

    Accepts arbitrary leading batch dimensions:
        q  : (..., Lq, d_model)
        kv : (..., Lk, d_kv)
        →  (..., Lq, d_model)

    Parameters
    ----------
    d_model   : feature dim of queries and of the output.
    num_heads : number of attention heads.
    d_kv      : feature dim of keys/values. Defaults to ``d_model``.
    d_head    : per-head dim. Defaults to ``d_model // num_heads`` (requires
                divisibility); pass an explicit value to decouple from ``d_model``.
    attn_dropout : dropout probability applied to the softmax attention weights.
    bias      : whether Q/K/V/out projections include biases.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
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

        self.d_model = d_model
        self.d_kv = d_kv
        self.num_heads = num_heads
        self.d_head = d_head
        self.inner_dim = num_heads * d_head
        self.attn_dropout = float(attn_dropout)
        self._scale = 1.0 / math.sqrt(d_head)

        self.q_proj = torch.nn.Linear(d_model, self.inner_dim, bias=bias)
        self.k_proj = torch.nn.Linear(d_kv, self.inner_dim, bias=bias)
        self.v_proj = torch.nn.Linear(d_kv, self.inner_dim, bias=bias)
        self.out_proj = torch.nn.Linear(self.inner_dim, d_model, bias=bias)

        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        torch.nn.init.xavier_uniform_(self.v_proj.weight)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        if bias:
            torch.nn.init.zeros_(self.q_proj.bias)
            torch.nn.init.zeros_(self.k_proj.bias)
            torch.nn.init.zeros_(self.v_proj.bias)
            torch.nn.init.zeros_(self.out_proj.bias)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        """(..., L, inner_dim) → (..., num_heads, L, d_head)"""
        return t.unflatten(-1, (self.num_heads, self.d_head)).transpose(-3, -2)

    def _merge(self, t: torch.Tensor) -> torch.Tensor:
        """(..., num_heads, L, d_head) → (..., L, inner_dim)"""
        return t.transpose(-3, -2).flatten(-2)

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
               If mask has one fewer dim than ``scores`` it is treated as
               head-agnostic and broadcast across the head dimension.
        returns (..., Lq, d_model)
        """
        q_h = self._split(self.q_proj(q))    # (..., H, Lq, D)
        k_h = self._split(self.k_proj(kv))   # (..., H, Lk, D)
        v_h = self._split(self.v_proj(kv))   # (..., H, Lk, D)

        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * self._scale  # (..., H, Lq, Lk)
        if mask is not None:
            if mask.dim() == scores.dim() - 1:
                mask = mask.unsqueeze(-3)
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        if self.attn_dropout > 0.0 and self.training:
            attn = torch.nn.functional.dropout(attn, p=self.attn_dropout)
        out = torch.matmul(attn, v_h)        # (..., H, Lq, D)
        out = self._merge(out)               # (..., Lq, inner_dim)
        return self.out_proj(out)            # (..., Lq, d_model)


__all__ = ["CrossAttention", "MultiHeadCrossAttention"]
