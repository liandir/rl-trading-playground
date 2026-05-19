"""Attention memory action value utilities for neural network architectures and reusable model components."""
from collections.abc import Callable
import torch

from src.network.core.attention import ResidualSelfAttentionBlock
from src.network.core.recurrent import RecurrentNetwork, _build_recurrent_cell
from src.network.core.vanilla import VanillaNetwork


class AttentionMemoryActionValueNetwork(torch.nn.Module):
    """
    Per-asset action-value network for multi-asset discrete action spaces.

    State layout (matches src.environment.generic.longshort):
        state_dim = d_global + num_assets * d_asset
        x[..., :d_global]                                 globals (time, cash_rel, rho, ...)
        x[..., d_global:].view(..., num_assets, d_asset)  per-asset block

    Architecture:
        input
          ├─ globals (d_global)              ──┐ broadcast across N tokens
          └─ assets  (N, d_asset)             ─┘
                                 │
                                 v
               per-asset MLP shared across N  →  (N, d_model)
                                 │
                                 + concat learned asset embedding
                                 v
                         (N, d_model + d_asset_emb)
                                 │
                                 v
               transformer attention+FFN stack × n_att_layers
               (optional LayerNorm)  →  tokens (N, d_model + d_asset_emb)
                                 │
                           flatten over N
                                 │
             ┌───────────────────┴───────────────────┐
             v                                       v
        recurrent module                     feedforward MLP
       (cell or stacked                       → (d_ff)
        RecurrentNetwork)
          → (d_mem)
             └───────────────────┬───────────────────┘
                                 v
                       combine: concat | add
                                 v
                       global trunk h
                       (d_mem + d_ff)  if combine_mode = "concat"
                       (d_mem)         if combine_mode = "add"   [requires d_mem == d_ff]
                                 │
        ┌────────────────────────┼────────────────────────┐
        v                        v                        v
    actor head             value head           per-asset head (per_asset mode)
    → action_dim          → (1) value           → (N, n_dir·K) per-token logits
                                                + global idle head → (1)
                                                scattered into env action layout

    ``output_mode`` selects how the actor head is built:

    * ``"pooled"`` (default): a single MLP maps the global trunk ``h`` to
      ``action_dim`` logits. Per-asset identity must be reconstructed from
      output position alone.
    * ``"per_asset"``: a shared per-asset MLP maps each post-attention token
      (concatenated with ``h`` broadcast) to ``n_directions·K`` logits, and a
      tiny global head emits the single idle logit. Outputs are scattered to
      match the env's ``[idle, longs(asset×bucket), shorts, closes]`` layout,
      which requires ``action_dim - 1 == n_directions·N·K`` for some integer
      bucket count ``K``. The bucket count is derived from ``action_dim``.

    Public interface matches ``ActionValueNetwork``/``RecurrentActionValueNetwork``:
        forward(x)       → (logits, values)          aux_horizons=None  [backwards-compatible]
                         → (logits, values, aux)     aux_horizons set
        forward_seq(x)   → same pattern as forward
        reset(B)         reset recurrent state
        get_states / set_states    state snapshot dict

    When ``aux_horizons`` is provided, three supervised prediction heads are
    attached to the trunk ``h`` and returned as a dict in position 3:
        reward    : (...)              one-step reward prediction
        asset_ret : (..., N, H)        per-asset log-return at each horizon
        asset_vol : (..., N, H)        per-asset volatility (softplus) at each horizon
    """

    name = "attention_memory_action_value"

    def __init__(
        self,
        num_assets: int,
        d_asset: int,
        d_global: int,
        action_dim: int,
        d_model: int = 64,
        asset_embed_dim: int | None = None,
        hidden_dims_asset=None,
        num_heads: int = 1,
        n_att_layers: int = 1,
        use_layer_norm: bool = False,
        d_mem: int = 64,
        d_ff: int = 64,
        hidden_dims_mem=None,
        hidden_dims_ff=None,
        hidden_dims_actor=None,
        hidden_dims_value=None,
        combine_mode: str = "concat",
        output_mode: str = "pooled",
        n_directions: int = 3,
        hidden_dims_per_asset_head=None,
        aux_horizons: tuple | None = None,
        hidden_dims_aux=None,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        recurrent_activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            num_assets (int): The num assets value.
            d_asset (int): The d asset value.
            d_global (int): The d global value.
            action_dim (int): The action dim value.
            d_model (int): The d model value. Defaults to ``64``.
            asset_embed_dim (int | None): The asset embed dim value. Defaults to ``None``.
            hidden_dims_asset (Any): The hidden dims asset value. Defaults to ``None``.
            num_heads (int): The num heads value. Defaults to ``1``.
            n_att_layers (int): The n att layers value. Defaults to ``1``.
            use_layer_norm (bool): The use layer norm value. Defaults to ``False``.
            d_mem (int): The d mem value. Defaults to ``64``.
            d_ff (int): The d ff value. Defaults to ``64``.
            hidden_dims_mem (Any): The hidden dims mem value. Defaults to ``None``.
            hidden_dims_ff (Any): The hidden dims ff value. Defaults to ``None``.
            hidden_dims_actor (Any): The hidden dims actor value. Defaults to ``None``.
            hidden_dims_value (Any): The hidden dims value value. Defaults to ``None``.
            combine_mode (str): The combine mode value. Defaults to ``'concat'``.
            output_mode (str): The output mode value. Defaults to ``'pooled'``.
            n_directions (int): The n directions value. Defaults to ``3``.
            hidden_dims_per_asset_head (Any): The hidden dims per asset head value. Defaults to ``None``.
            aux_horizons (tuple | None): The aux horizons value. Defaults to ``None``.
            hidden_dims_aux (Any): The hidden dims aux value. Defaults to ``None``.
            activation (Callable[[torch.Tensor], torch.Tensor]): The activation value. Defaults to ``torch.tanh``.
            recurrent_activation (Callable[[torch.Tensor], torch.Tensor]): The recurrent activation value. Defaults to ``torch.tanh``.
            recurrent_type (str): The recurrent type value. Defaults to ``'simple'``.
            recurrent_kwargs (dict | None): The recurrent kwargs value. Defaults to ``None``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__()
        if output_mode not in ("pooled", "per_asset"):
            raise ValueError(f"output_mode must be 'pooled' or 'per_asset', got '{output_mode}'.")
        if combine_mode not in ("concat", "add"):
            raise ValueError(f"combine_mode must be 'concat' or 'add', got '{combine_mode}'.")
        if combine_mode == "add" and int(d_mem) != int(d_ff):
            raise ValueError(
                f"combine_mode='add' requires d_mem == d_ff, got d_mem={d_mem}, d_ff={d_ff}."
            )

        hidden_dims_asset = hidden_dims_asset or []
        hidden_dims_mem = hidden_dims_mem or []
        hidden_dims_ff = hidden_dims_ff or []
        hidden_dims_actor = hidden_dims_actor or []
        hidden_dims_value = hidden_dims_value or []
        hidden_dims_per_asset_head = hidden_dims_per_asset_head or []
        hidden_dims_aux = hidden_dims_aux or []

        self.num_assets = int(num_assets)
        self.d_asset = int(d_asset)
        self.d_global = int(d_global)
        self.state_dim = self.d_global + self.num_assets * self.d_asset
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.asset_embed_dim = self.d_model if asset_embed_dim is None else int(asset_embed_dim)
        self.num_heads = int(num_heads)
        self.n_att_layers = int(n_att_layers)
        self.use_layer_norm = bool(use_layer_norm)
        self.d_mem = int(d_mem)
        self.d_ff = int(d_ff)
        self.combine_mode = combine_mode
        self.output_mode = output_mode
        self.n_directions = int(n_directions)
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.recurrent_type = recurrent_type
        self.recurrent_kwargs = dict(recurrent_kwargs or {})
        if self.asset_embed_dim <= 0:
            raise ValueError(f"asset_embed_dim must be >= 1, got {self.asset_embed_dim}.")
        if self.n_att_layers < 1:
            raise ValueError(f"n_att_layers must be >= 1, got {self.n_att_layers}.")
        self.token_dim = self.d_model + self.asset_embed_dim
        if self.num_heads > 1 and self.token_dim % self.num_heads != 0:
            raise ValueError(
                f"d_model + asset_embed_dim ({self.token_dim}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )

        # Per-asset feature extractor: sees [asset_i, globals_broadcast]
        self.asset_extractor = VanillaNetwork(
            self.d_asset + self.d_global,
            self.d_model,
            hidden_dims=hidden_dims_asset,
            activation=activation,
        )
        self.asset_embedding = torch.nn.Embedding(self.num_assets, self.asset_embed_dim)
        torch.nn.init.xavier_uniform_(self.asset_embedding.weight)
        self.register_buffer("_asset_ids", torch.arange(self.num_assets, dtype=torch.long), persistent=False)

        # Transformer-style self-attention stack over concatenated asset tokens.
        self.attention_layers = torch.nn.ModuleList(
            ResidualSelfAttentionBlock(self.token_dim, self.num_heads, use_layer_norm=self.use_layer_norm)
            for _ in range(self.n_att_layers)
        )

        # Dim fed into the recurrent and feedforward branches
        self._post_attn_dim = self.num_assets * self.token_dim

        # Recurrent branch (memory). With hidden_dims_mem == [] this is a single
        # recurrent cell mapping post_attn_dim → d_mem directly. With non-empty
        # hidden_dims_mem it's a stacked RecurrentNetwork with a final Linear
        # projection to d_mem.
        if hidden_dims_mem:
            self.recurrent = RecurrentNetwork(
                self._post_attn_dim,
                self.d_mem,
                hidden_dims=hidden_dims_mem,
                activation=recurrent_activation,
                recurrent_type=recurrent_type,
                recurrent_kwargs=self.recurrent_kwargs,
            )
        else:
            self.recurrent = _build_recurrent_cell(
                self._post_attn_dim,
                self.d_mem,
                recurrent_activation,
                recurrent_type,
                self.recurrent_kwargs,
            )
        self._recurrent_is_network = isinstance(self.recurrent, RecurrentNetwork)

        # Feedforward branch (parallel to recurrent)
        self.feedforward = VanillaNetwork(
            self._post_attn_dim,
            self.d_ff,
            hidden_dims=hidden_dims_ff,
            activation=activation,
        )

        head_in = self.d_mem if self.combine_mode == "add" else self.d_mem + self.d_ff
        self.value = VanillaNetwork(head_in, 1, hidden_dims=hidden_dims_value, activation=activation)

        if self.output_mode == "pooled":
            self.actor = VanillaNetwork(head_in, self.action_dim, hidden_dims=hidden_dims_actor, activation=activation)
            self.n_size_buckets = None
            self.per_asset_head = None
            self.idle_head = None
        else:  # per_asset
            per_asset_logits = self.action_dim - 1
            denom = self.n_directions * self.num_assets
            if per_asset_logits <= 0 or per_asset_logits % denom != 0:
                raise ValueError(
                    f"output_mode='per_asset' requires action_dim - 1 ({per_asset_logits}) "
                    f"divisible by n_directions * num_assets ({denom})."
                )
            self.n_size_buckets = per_asset_logits // denom
            per_asset_in = self.token_dim + head_in
            per_asset_out = self.n_directions * self.n_size_buckets
            self.per_asset_head = VanillaNetwork(
                per_asset_in,
                per_asset_out,
                hidden_dims=hidden_dims_per_asset_head,
                activation=activation,
            )
            self.idle_head = VanillaNetwork(head_in, 1, hidden_dims=hidden_dims_actor, activation=activation)
            self.actor = None

        # Optional auxiliary prediction heads
        if aux_horizons is not None:
            self.aux_horizons = tuple(int(h) for h in aux_horizons)
            if len(self.aux_horizons) < 1:
                raise ValueError("aux_horizons must contain at least one horizon.")
            if any(h <= 0 for h in self.aux_horizons):
                raise ValueError(f"aux_horizons must be positive, got {self.aux_horizons}.")
            self.n_aux_horizons = len(self.aux_horizons)
            self.aux_reward    = VanillaNetwork(head_in, 1, hidden_dims=hidden_dims_aux, activation=activation)
            self.aux_asset_ret = VanillaNetwork(head_in, self.num_assets * self.n_aux_horizons, hidden_dims=hidden_dims_aux, activation=activation)
            self.aux_asset_vol = VanillaNetwork(head_in, self.num_assets * self.n_aux_horizons, hidden_dims=hidden_dims_aux, activation=activation)
        else:
            self.aux_horizons  = None
            self.n_aux_horizons = 0
            self.aux_reward = self.aux_asset_ret = self.aux_asset_vol = None

    def _encode_tokens(self, x_flat: torch.Tensor) -> torch.Tensor:
        """x_flat : (B, state_dim)  →  tokens (B, N, token_dim)

        Splits into globals + per-asset block, runs the shared per-asset MLP,
        concatenates a learned per-asset embedding, and applies the
        transformer-style attention/FFN stack across assets. Flattening is left to
        ``_flatten_tokens`` so callers can also access the per-token
        representation.

        Args:
            x_flat (torch.Tensor): The x flat value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        B = x_flat.shape[0]
        globals_feat = x_flat[:, : self.d_global]                                          # (B, d_global)
        asset_feat = x_flat[:, self.d_global :].view(B, self.num_assets, self.d_asset)     # (B, N, d_asset)

        globals_broadcast = globals_feat.unsqueeze(1).expand(-1, self.num_assets, -1)      # (B, N, d_global)
        per_asset_in = torch.cat([asset_feat, globals_broadcast], dim=-1)                   # (B, N, d_asset+d_global)

        asset_repr = self.asset_extractor(per_asset_in).view(B, self.num_assets, self.d_model)
        asset_emb = self.asset_embedding(self._asset_ids).unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([asset_repr, asset_emb], dim=-1)                                 # (B, N, token_dim)
        for block in self.attention_layers:
            tokens = block(tokens)
        return tokens

    def _flatten_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, N, token_dim) → (B, post_attn_dim)

        Args:
            tokens (torch.Tensor): The tokens value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return tokens.reshape(tokens.shape[0], -1)

    def _build_logits(self, tokens: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """tokens : (B, N, token_dim)
        h      : (B, d_mem + d_ff)
        returns logits (B, action_dim)

        Args:
            tokens (torch.Tensor): The tokens value.
            h (torch.Tensor): The h value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        if self.output_mode == "pooled":
            return self.actor(h)

        B = h.shape[0]
        h_broadcast = h.unsqueeze(1).expand(-1, self.num_assets, -1)                        # (B, N, head_in)
        per_asset_in = torch.cat([tokens, h_broadcast], dim=-1)                             # (B, N, token_dim+head_in)
        per_asset_logits = self.per_asset_head(per_asset_in)                                # (B, N, n_dir*K)
        per_asset_logits = per_asset_logits.view(B, self.num_assets, self.n_directions, self.n_size_buckets)
        # Direction-major scatter: [idle, dir0(asset×bucket), dir1(...), ...]
        parts = [self.idle_head(h)]                                                         # (B, 1)
        for d in range(self.n_directions):
            parts.append(per_asset_logits[:, :, d, :].reshape(B, -1))                       # (B, N*K)
        return torch.cat(parts, dim=-1)                                                     # (B, action_dim)

    def _combine(self, h_mem: torch.Tensor, h_ff: torch.Tensor) -> torch.Tensor:
        """Combine for AttentionMemoryActionValueNetwork.

        Args:
            h_mem (torch.Tensor): The h mem value.
            h_ff (torch.Tensor): The h ff value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        if self.combine_mode == "add":
            return h_mem + h_ff
        return torch.cat([h_mem, h_ff], dim=-1)

    def _build_aux(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """h : (M, head_in) → flat aux dict with leading dim M.

        Args:
            h (torch.Tensor): The h value.

        Returns:
            dict[str, torch.Tensor]: The computed or requested result.
        """
        return {
            "reward":    self.aux_reward(h).squeeze(-1),
            "asset_ret": self.aux_asset_ret(h).view(h.shape[0], self.num_assets, self.n_aux_horizons),
            "asset_vol": torch.nn.functional.softplus(
                self.aux_asset_vol(h).view(h.shape[0], self.num_assets, self.n_aux_horizons)
            ),
        }

    def reset(self, batch_size: int = 1) -> None:
        """Reset internal state for a new episode or stream.

        Args:
            batch_size (int): The batch size value. Defaults to ``1``.

        Returns:
            None: This function does not return a value.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.recurrent.reset(batch_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        """Compute the forward pass.

        Args:
            x (torch.Tensor): The x value.

        Returns:
            Any: The computed or requested result.
        """
        shape = x.shape  # (..., state_dim)
        x_flat = x.reshape(-1, self.state_dim)                          # (B', state_dim)
        tokens = self._encode_tokens(x_flat)                            # (B', N, token_dim)
        z = self._flatten_tokens(tokens)                                # (B', post_attn_dim)

        h_mem = self.recurrent(z)                                       # (B', d_mem)
        h_ff = self.feedforward(z)                                      # (B', d_ff)
        h = self._combine(h_mem, h_ff)                                  # (B', head_in)

        logits_flat = self._build_logits(tokens, h)                     # (B', action_dim)
        values_flat = self.value(h).squeeze(-1)                         # (B',)

        logits = logits_flat.reshape(shape[:-1] + (self.action_dim,))
        values = values_flat.reshape(shape[:-1])

        if self.aux_horizons is None:
            return logits, values

        aux_flat = self._build_aux(h)
        aux = {
            "reward":    aux_flat["reward"].reshape(shape[:-1]),
            "asset_ret": aux_flat["asset_ret"].reshape(shape[:-1] + (self.num_assets, self.n_aux_horizons)),
            "asset_vol": aux_flat["asset_vol"].reshape(shape[:-1] + (self.num_assets, self.n_aux_horizons)),
        }
        return logits, values, aux

    def forward_seq(self, x_seq: torch.Tensor):
        """x_seq : (T, state_dim) or (T, B, state_dim)

        Returns (logits, values) when aux_horizons is None, otherwise
        (logits, values, aux) — matching the AuxiliaryPerAssetActionValueNetwork
        interface so this network can be dropped in for HierarchicalAuxAACAgent.

        Args:
            x_seq (torch.Tensor): The x seq value.

        Returns:
            Any: The computed or requested result.
        """
        if x_seq.dim() == 2:
            if x_seq.shape[-1] != self.state_dim:
                raise ValueError(f"Expected last dim {self.state_dim}, got {x_seq.shape[-1]}.")
            T = x_seq.shape[0]
            B = 1
            x_batched = x_seq.unsqueeze(1)  # (T, 1, state_dim)
            squeeze_batch = True
        elif x_seq.dim() == 3:
            if x_seq.shape[-1] != self.state_dim:
                raise ValueError(f"Expected last dim {self.state_dim}, got {x_seq.shape[-1]}.")
            T, B, _ = x_seq.shape
            x_batched = x_seq
            squeeze_batch = False
        else:
            raise ValueError(
                f"Expected sequence input with shape (T, {self.state_dim}) or "
                f"(T, B, {self.state_dim}), got {tuple(x_seq.shape)}."
            )

        # Encode all (T, B) tokens in a single batched pass
        x_flat = x_batched.reshape(T * B, self.state_dim)
        tokens_flat = self._encode_tokens(x_flat)                    # (T*B, N, token_dim)
        z_flat = self._flatten_tokens(tokens_flat)                   # (T*B, post_attn_dim)
        z_seq = z_flat.view(T, B, -1)                                # (T, B, post_attn_dim)

        # Recurrent branch: sequence mode
        h_mem_seq = self.recurrent.forward_seq(z_seq)                # (T, B, d_mem)

        # Feedforward branch: per-timestep, stateless
        h_ff_seq = self.feedforward(z_flat).view(T, B, self.d_ff)    # (T, B, d_ff)

        h_seq = self._combine(h_mem_seq, h_ff_seq)                   # (T, B, head_in)
        h_flat = h_seq.reshape(T * B, -1)

        logits_flat = self._build_logits(tokens_flat, h_flat)        # (T*B, action_dim)
        values_flat = self.value(h_flat).squeeze(-1)                 # (T*B,)

        logits_seq = logits_flat.view(T, B, self.action_dim)
        values_seq = values_flat.view(T, B)

        if squeeze_batch:
            logits_seq = logits_seq.squeeze(1)                       # (T, action_dim)
            values_seq = values_seq.squeeze(1)                       # (T,)

        if self.aux_horizons is None:
            return logits_seq, values_seq

        aux_flat = self._build_aux(h_flat)
        aux_seq = {
            "reward":    aux_flat["reward"].view(T, B),
            "asset_ret": aux_flat["asset_ret"].view(T, B, self.num_assets, self.n_aux_horizons),
            "asset_vol": aux_flat["asset_vol"].view(T, B, self.num_assets, self.n_aux_horizons),
        }
        if squeeze_batch:
            aux_seq = {k: v.squeeze(1) for k, v in aux_seq.items()}
        return logits_seq, values_seq, aux_seq

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        """Return a snapshot of recurrent states.

        Args:
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.

        Returns:
            dict: The computed or requested result.
        """
        if self._recurrent_is_network:
            rec_state = self.recurrent.get_states(clone=clone, detach=detach)
        else:
            rec_state = self.recurrent.get_state(clone=clone, detach=detach)
        return {"recurrent": rec_state}

    def set_states(self, states: dict, clone: bool = True, detach: bool = True, strict: bool = True) -> None:
        """Restore recurrent states from a snapshot.

        Args:
            states (dict): The states value.
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.
            strict (bool): The strict value. Defaults to ``True``.

        Returns:
            None: This function does not return a value.
        """
        if strict and "recurrent" not in states:
            raise KeyError("Missing key 'recurrent' in states snapshot.")
        rec_state = states.get("recurrent")
        if self._recurrent_is_network:
            self.recurrent.set_states(rec_state, clone=clone, detach=detach, strict=strict)
        else:
            self.recurrent.set_state(rec_state, clone=clone, detach=detach)

