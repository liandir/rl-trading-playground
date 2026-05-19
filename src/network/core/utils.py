"""Generic network building-block utilities shared across attention and positional modules."""
import math

import torch


# ---------------------------------------------------------------------------
# attention primitives
# ---------------------------------------------------------------------------

def _split_heads(t: torch.Tensor, num_heads: int, d_head: int) -> torch.Tensor:
    """Reshape ``(..., L, inner_dim)`` into ``(..., num_heads, L, d_head)``.

    Args:
        t (torch.Tensor): Tensor whose last axis carries the fused per-head
            features, i.e. shape ``(..., L, num_heads * d_head)``.
        num_heads (int): Number of attention heads.
        d_head (int): Per-head feature size; must satisfy
            ``num_heads * d_head == t.shape[-1]``.

    Returns:
        torch.Tensor: Tensor with heads pulled out into their own axis, shape
            ``(..., num_heads, L, d_head)``.
    """
    return t.unflatten(-1, (num_heads, d_head)).transpose(-3, -2)


def _merge_heads(t: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_split_heads`: collapse the head axis back into features.

    Args:
        t (torch.Tensor): Per-head tensor of shape
            ``(..., num_heads, L, d_head)``.

    Returns:
        torch.Tensor: Tensor with heads merged into the feature axis, shape
            ``(..., L, num_heads * d_head)``.
    """
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
    """Run scaled dot-product attention through PyTorch's fused SDPA kernel.

    Args:
        q_h (torch.Tensor): Per-head queries, shape
            ``(..., num_heads, L_q, d_head)``.
        k_h (torch.Tensor): Per-head keys, shape
            ``(..., num_heads, L_k, d_head)``.
        v_h (torch.Tensor): Per-head values, shape
            ``(..., num_heads, L_k, d_v)``.
        scale (float): Multiplicative softmax scale applied before exponentiating
            the logits (typically ``1 / sqrt(d_head)``).
        mask (torch.Tensor | None): Attention mask broadcastable to
            ``(..., L_q, L_k)``. A leading head axis is added automatically when
            the mask is one rank short. ``None`` disables masking.
        attn_dropout (float): Dropout probability applied to the attention
            weights when ``training`` is ``True``.
        training (bool): When ``False`` the dropout probability is ignored
            (eval-mode behaviour).

    Returns:
        torch.Tensor: Attention output, shape ``(..., num_heads, L_q, d_v)``.
    """
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
    """Initialize a linear layer with Xavier-uniform weights and zeroed bias.

    Args:
        linear (torch.nn.Linear): Layer to initialize in place.
        bias (bool): Whether ``linear`` has a learned bias term that should be
            zero-initialized.
    """
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
    """Build the standard transformer position-wise feed-forward sub-block.

    The block is ``Linear → GELU → Dropout → Linear → Dropout`` and its
    ``nn.Linear`` layers are initialized via :func:`_init_linear`.

    Args:
        d_model (int): Input and output feature dimension.
        ff_hidden_dim (int): Width of the inner hidden layer.
        ff_dropout (float): Dropout probability applied after both linear
            projections.
        bias (bool): Whether the linear projections include a bias term.

    Returns:
        torch.nn.Sequential: The assembled feed-forward module.
    """
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
    """Build a fixed sinusoidal positional encoding of shape ``(T, d)``.

    Position ``0`` represents the oldest token and position ``T - 1`` the
    newest, matching the temporal convention used throughout the codebase.

    Args:
        T (int): Sequence length / number of positions.
        d (int): Encoding feature dimension.
        device (torch.device): Device on which to allocate the encoding.
        dtype (torch.dtype): Floating-point dtype of the encoding.

    Returns:
        torch.Tensor: Positional encoding of shape ``(T, d)`` with even
            channels carrying sine components and odd channels cosine
            components.
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
    """Build a boolean causal (and optionally windowed) attention mask.

    ``mask[t, s]`` is ``True`` (attend) iff ``s <= t`` (causality) and
    ``t - s < W`` (sliding window). When ``W is None`` or ``W >= T`` the mask
    is purely causal.

    Args:
        T (int): Sequence length, i.e. mask side length.
        W (int | None): Sliding-window size in tokens. ``None`` (or any value
            ``>= T``) disables the window and keeps the mask fully causal.
        device (torch.device): Device on which to build the mask.

    Returns:
        torch.Tensor: Boolean mask of shape ``(T, T)`` where ``True`` means the
            query at row ``t`` should attend to the key at column ``s``.
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
