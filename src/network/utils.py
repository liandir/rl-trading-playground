"""Generic network building-block utilities shared across attention and positional modules."""
import math

import torch


# ---------------------------------------------------------------------------
# attention primitives
# ---------------------------------------------------------------------------

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
    if mask is not None and mask.dim() == q_h.dim() - 1:
        mask = mask.unsqueeze(-3)
    return torch.nn.functional.scaled_dot_product_attention(
        q_h,
        k_h,
        v_h,
        attn_mask=mask,
        dropout_p=attn_dropout if training else 0.0,
        scale=scale,
    )


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


# ---------------------------------------------------------------------------
# positional encoding
# ---------------------------------------------------------------------------

def _sinusoidal_pos_encoding(T: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Sinusoidal positional encoding, shape (T, d).
    Position 0 = oldest token, T-1 = newest token.
    """
    pos   = torch.arange(T, dtype=dtype, device=device).unsqueeze(1)   # (T, 1)
    n_sin = (d + 1) // 2
    n_cos = d // 2
    div   = torch.exp(torch.arange(n_sin, dtype=dtype, device=device) * -(math.log(10000.0) / max(d, 1)))
    enc   = torch.zeros(T, d, dtype=dtype, device=device)
    enc[:, 0::2] = torch.sin(pos * div)
    if n_cos > 0:
        enc[:, 1::2] = torch.cos(pos * div[:n_cos])
    return enc


# ---------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------

def _build_causal_mask(T: int, W: int | None, device: torch.device) -> torch.Tensor:
    """
    Bool causal mask of shape (T, T). True = attend, False = mask out.
    mask[t, s] = True iff s <= t (causal) and t - s < W (window cap, when W < T).
    """
    idx  = torch.arange(T, device=device)
    row  = idx.unsqueeze(1)   # (T, 1)
    col  = idx.unsqueeze(0)   # (1, T)
    mask = col <= row
    if W is not None and W < T:
        mask = mask & (row - col < W)
    return mask


__all__ = [
    "_build_causal_mask",
    "_build_transformer_ffn",
    "_init_linear",
    "_merge_heads",
    "_scaled_dot_product_attention",
    "_sinusoidal_pos_encoding",
    "_split_heads",
]