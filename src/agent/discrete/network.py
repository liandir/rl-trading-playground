import torch

from src.network.attention import CrossAttention, MultiHeadCrossAttention
from src.network.vanilla import VanillaNetwork
from src.network.recurrent import RecurrentNetwork, _build_recurrent_cell


class ActionValueNetwork(torch.nn.Module):
    """Shared feedforward trunk with categorical-policy and scalar-value heads."""

    name = "action_value"

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=None,
        hidden_dims_actor=None,
        hidden_dims_value=None,
        activation=torch.relu,
    ):
        super().__init__()
        hidden_dims = hidden_dims or []
        hidden_dims_actor = hidden_dims_actor or []
        hidden_dims_value = hidden_dims_value or []

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = activation

        self.layers = torch.nn.ModuleList()
        prev = state_dim
        for h in hidden_dims:
            self.layers.append(torch.nn.Linear(prev, h))
            prev = h

        self.actor = VanillaNetwork(prev, action_dim, hidden_dims=hidden_dims_actor, activation=activation)
        self.value = VanillaNetwork(prev, 1, hidden_dims=hidden_dims_value, activation=activation)

    def forward(self, x):
        shape = x.shape  # (..., state_dim)
        z = x.reshape(-1, self.state_dim)
        for layer in self.layers:
            z = self.activation(layer(z))
        logits = self.actor(z).view(*shape[:-1], self.action_dim)
        values = self.value(z).squeeze(-1).view(*shape[:-1])
        return logits, values


class RecurrentActionValueNetwork(torch.nn.Module):
    """Shared recurrent trunk with categorical-policy and scalar-value heads."""

    name = "recurrent_action_value"

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=None,
        hidden_dims_actor=None,
        hidden_dims_value=None,
        activation=torch.tanh,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or []
        hidden_dims_actor = hidden_dims_actor or []
        hidden_dims_value = hidden_dims_value or []

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = activation
        self.recurrent_type = recurrent_type
        self.recurrent_kwargs = dict(recurrent_kwargs or {})

        self.layers = torch.nn.ModuleList()
        prev = state_dim
        for h in hidden_dims:
            self.layers.append(_build_recurrent_cell(prev, h, activation, recurrent_type, self.recurrent_kwargs))
            prev = h

        self.actor = RecurrentNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_actor,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=self.recurrent_kwargs,
        )
        self.value = RecurrentNetwork(
            prev,
            1,
            hidden_dims=hidden_dims_value,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=self.recurrent_kwargs,
        )

    def reset(self, batch_size: int = 1):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        for layer in self.layers:
            layer.reset(batch_size, device=device, dtype=dtype)
        self.actor.reset(batch_size, device=device, dtype=dtype)
        self.value.reset(batch_size, device=device, dtype=dtype)

    def forward(self, x):
        z = x.reshape(-1, self.state_dim)
        for layer in self.layers:
            z = layer(z)
        logits = self.actor(z).view(-1, self.action_dim)
        values = self.value(z).view(-1)
        return logits, values

    def forward_seq(self, x_seq):
        z_seq = x_seq
        for layer in self.layers:
            z_seq = layer.forward_seq(z_seq)
        logits = self.actor.forward_seq(z_seq)  # (T, [B,] A)
        values = self.value.forward_seq(z_seq).squeeze(-1)  # (T, [B,])
        return logits, values

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        trunk = [cell.get_state(clone=clone, detach=detach) for cell in self.layers]
        actor = self.actor.get_states(clone=clone, detach=detach)
        value = self.value.get_states(clone=clone, detach=detach)
        return {"trunk": trunk, "actor": actor, "v": value}

    def set_states(self, states: dict, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict:
            for key in ("trunk", "actor", "v"):
                if key not in states:
                    raise KeyError(f"Missing key '{key}' in states snapshot.")

        trunk_states = states.get("trunk", [])
        actor_states = states.get("actor", [])
        v_states = states.get("v", [])

        if strict and len(trunk_states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} trunk states, got {len(trunk_states)}.")

        for cell, state in zip(self.layers, trunk_states):
            cell.set_state(state, clone=clone, detach=detach)

        self.actor.set_states(actor_states, clone=clone, detach=detach, strict=strict)
        self.value.set_states(v_states, clone=clone, detach=detach, strict=strict)


class ActionQNetwork(torch.nn.Module):
    """Shared feedforward trunk with categorical-policy and discrete-Q heads."""

    name = "action_q"

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=[],
        hidden_dims_actor=[],
        hidden_dims_q=[],
        activation=torch.relu,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = activation

        self.layers = torch.nn.ModuleList()
        prev = state_dim
        for h in hidden_dims:
            self.layers.append(torch.nn.Linear(prev, h))
            prev = h

        self.actor = VanillaNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_actor,
            activation=activation,
        )

        self.q_head = VanillaNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_q,
            activation=activation,
        )

    def forward(self, x):
        z = x.view(-1, self.state_dim)
        for layer in self.layers:
            z = self.activation(layer(z))
        return self.actor(z), self.q_head(z)


class RecurrentActionQNetwork(torch.nn.Module):
    """Shared recurrent trunk with categorical-policy and discrete-Q heads."""

    name = "recurrent_action_q"

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=None,
        hidden_dims_actor=None,
        hidden_dims_q=None,
        activation=torch.tanh,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or []
        hidden_dims_actor = hidden_dims_actor or []
        hidden_dims_q = hidden_dims_q or []

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = activation
        self.recurrent_type = recurrent_type
        self.recurrent_kwargs = dict(recurrent_kwargs or {})

        self.layers = torch.nn.ModuleList()
        prev = state_dim
        for h in hidden_dims:
            self.layers.append(_build_recurrent_cell(prev, h, activation, recurrent_type, self.recurrent_kwargs))
            prev = h

        self.actor = RecurrentNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_actor,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=self.recurrent_kwargs,
        )
        self.q_head = RecurrentNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_q,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=self.recurrent_kwargs,
        )

    def reset(self, batch_size: int = 1):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        for layer in self.layers:
            layer.reset(batch_size, device=device, dtype=dtype)
        self.actor.reset(batch_size, device=device, dtype=dtype)
        self.q_head.reset(batch_size, device=device, dtype=dtype)

    def forward(self, x):
        z = x.view(-1, self.state_dim)
        for layer in self.layers:
            z = layer(z)
        return self.actor(z), self.q_head(z)

    def forward_seq(self, x_seq):
        z_seq = x_seq
        for layer in self.layers:
            z_seq = layer.forward_seq(z_seq)
        return self.actor.forward_seq(z_seq), self.q_head.forward_seq(z_seq)

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        trunk = [cell.get_state(clone=clone, detach=detach) for cell in self.layers]
        actor = self.actor.get_states(clone=clone, detach=detach)
        q_head = self.q_head.get_states(clone=clone, detach=detach)
        return {"trunk": trunk, "actor": actor, "q": q_head}

    def set_states(self, states: dict, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict:
            for key in ("trunk", "actor", "q"):
                if key not in states:
                    raise KeyError(f"Missing key '{key}' in states snapshot.")

        trunk_states = states.get("trunk", [])
        actor_states = states.get("actor", [])
        q_states = states.get("q", [])

        if strict and len(trunk_states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} trunk states, got {len(trunk_states)}.")

        for cell, state in zip(self.layers, trunk_states):
            cell.set_state(state, clone=clone, detach=detach)

        self.actor.set_states(actor_states, clone=clone, detach=detach, strict=strict)
        self.q_head.set_states(q_states, clone=clone, detach=detach, strict=strict)


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
                                 v
               self-attention across N tokens
               (optional LayerNorm + residual)  →  tokens (N, d_model)
                                 │
                   optional mean-pool / concat over N
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
      output position alone — works, but information-lossy when the global
      trunk was mean-pooled.
    * ``"per_asset"``: a shared per-asset MLP maps each post-attention token
      (concatenated with ``h`` broadcast) to ``n_directions·K`` logits, and a
      tiny global head emits the single idle logit. Outputs are scattered to
      match the env's ``[idle, longs(asset×bucket), shorts, closes]`` layout,
      which requires ``action_dim - 1 == n_directions·N·K`` for some integer
      bucket count ``K``. The bucket count is derived from ``action_dim``.

    Public interface matches ``ActionValueNetwork``/``RecurrentActionValueNetwork``:
        forward(x)       → (logits, values)
        forward_seq(x)   → (logits_seq, values_seq)
        reset(B)         reset recurrent state
        get_states / set_states    state snapshot dict
    """

    name = "attention_memory_action_value"

    def __init__(
        self,
        num_assets: int,
        d_asset: int,
        d_global: int,
        action_dim: int,
        d_model: int = 64,
        hidden_dims_asset=None,
        num_heads: int = 1,
        use_layer_norm: bool = False,
        use_mean_pool: bool = True,
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
        activation=torch.tanh,
        recurrent_activation=torch.tanh,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
    ):
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

        self.num_assets = int(num_assets)
        self.d_asset = int(d_asset)
        self.d_global = int(d_global)
        self.state_dim = self.d_global + self.num_assets * self.d_asset
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.use_layer_norm = bool(use_layer_norm)
        self.use_mean_pool = bool(use_mean_pool)
        self.d_mem = int(d_mem)
        self.d_ff = int(d_ff)
        self.combine_mode = combine_mode
        self.output_mode = output_mode
        self.n_directions = int(n_directions)
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.recurrent_type = recurrent_type
        self.recurrent_kwargs = dict(recurrent_kwargs or {})

        # Per-asset feature extractor: sees [asset_i, globals_broadcast]
        self.asset_extractor = VanillaNetwork(
            self.d_asset + self.d_global,
            self.d_model,
            hidden_dims=hidden_dims_asset,
            activation=activation,
        )

        # Cross-attention across asset tokens (self-attention)
        if self.num_heads <= 1:
            self.attention = CrossAttention(self.d_model)
        else:
            self.attention = MultiHeadCrossAttention(self.d_model, num_heads=self.num_heads)

        self.ln_attn = torch.nn.LayerNorm(self.d_model) if self.use_layer_norm else torch.nn.Identity()

        # Dim fed into the recurrent and feedforward branches
        if self.use_mean_pool:
            self._post_attn_dim = self.d_model
        else:
            self._post_attn_dim = self.num_assets * self.d_model

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
            per_asset_in = self.d_model + head_in
            per_asset_out = self.n_directions * self.n_size_buckets
            self.per_asset_head = VanillaNetwork(
                per_asset_in,
                per_asset_out,
                hidden_dims=hidden_dims_per_asset_head,
                activation=activation,
            )
            self.idle_head = VanillaNetwork(head_in, 1, hidden_dims=hidden_dims_actor, activation=activation)
            self.actor = None

    def _encode_tokens(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat : (B, state_dim)  →  tokens (B, N, d_model)

        Splits into globals + per-asset block, runs the shared per-asset MLP,
        and applies cross-attention across assets (with optional residual + LN).
        Pooling is left to ``_pool_tokens`` so callers can also access the
        per-token representation.
        """
        B = x_flat.shape[0]
        globals_feat = x_flat[:, : self.d_global]                                          # (B, d_global)
        asset_feat = x_flat[:, self.d_global :].view(B, self.num_assets, self.d_asset)     # (B, N, d_asset)

        globals_broadcast = globals_feat.unsqueeze(1).expand(-1, self.num_assets, -1)      # (B, N, d_global)
        per_asset_in = torch.cat([asset_feat, globals_broadcast], dim=-1)                   # (B, N, d_asset+d_global)

        tokens = self.asset_extractor(per_asset_in).view(B, self.num_assets, self.d_model)  # (B, N, d_model)
        attn_out = self.attention(tokens, tokens)                                           # (B, N, d_model)
        return self.ln_attn(tokens + attn_out)                                              # (B, N, d_model)

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, N, d_model) → (B, post_attn_dim)"""
        if self.use_mean_pool:
            return tokens.mean(dim=1)
        return tokens.reshape(tokens.shape[0], -1)

    def _build_logits(self, tokens: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, N, d_model)
        h      : (B, d_mem + d_ff)
        returns logits (B, action_dim)
        """
        if self.output_mode == "pooled":
            return self.actor(h)

        B = h.shape[0]
        h_broadcast = h.unsqueeze(1).expand(-1, self.num_assets, -1)                        # (B, N, head_in)
        per_asset_in = torch.cat([tokens, h_broadcast], dim=-1)                             # (B, N, d_model+head_in)
        per_asset_logits = self.per_asset_head(per_asset_in)                                # (B, N, n_dir*K)
        per_asset_logits = per_asset_logits.view(B, self.num_assets, self.n_directions, self.n_size_buckets)
        # Direction-major scatter: [idle, dir0(asset×bucket), dir1(...), ...]
        parts = [self.idle_head(h)]                                                         # (B, 1)
        for d in range(self.n_directions):
            parts.append(per_asset_logits[:, :, d, :].reshape(B, -1))                       # (B, N*K)
        return torch.cat(parts, dim=-1)                                                     # (B, action_dim)

    def _combine(self, h_mem: torch.Tensor, h_ff: torch.Tensor) -> torch.Tensor:
        if self.combine_mode == "add":
            return h_mem + h_ff
        return torch.cat([h_mem, h_ff], dim=-1)

    def reset(self, batch_size: int = 1):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.recurrent.reset(batch_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape  # (..., state_dim)
        x_flat = x.reshape(-1, self.state_dim)                          # (B', state_dim)
        tokens = self._encode_tokens(x_flat)                            # (B', N, d_model)
        z = self._pool_tokens(tokens)                                   # (B', post_attn_dim)

        h_mem = self.recurrent(z)                                       # (B', d_mem)
        h_ff = self.feedforward(z)                                      # (B', d_ff)
        h = self._combine(h_mem, h_ff)                                  # (B', head_in)

        logits_flat = self._build_logits(tokens, h)                     # (B', action_dim)
        values_flat = self.value(h).squeeze(-1)                         # (B',)

        logits = logits_flat.reshape(shape[:-1] + (self.action_dim,))
        values = values_flat.reshape(shape[:-1])
        return logits, values

    def forward_seq(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_seq : (T, state_dim) or (T, B, state_dim)

        Encodes the per-asset/attention stack per-timestep, runs the recurrent
        cell over the full sequence via ``forward_seq`` (CuDNN fused path for
        GRU), runs the feedforward branch per-timestep, concatenates, and
        passes through the actor/value heads.
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
        tokens_flat = self._encode_tokens(x_flat)                    # (T*B, N, d_model)
        z_flat = self._pool_tokens(tokens_flat)                      # (T*B, post_attn_dim)
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
        return logits_seq, values_seq

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        if self._recurrent_is_network:
            rec_state = self.recurrent.get_states(clone=clone, detach=detach)
        else:
            rec_state = self.recurrent.get_state(clone=clone, detach=detach)
        return {"recurrent": rec_state}

    def set_states(self, states: dict, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict and "recurrent" not in states:
            raise KeyError("Missing key 'recurrent' in states snapshot.")
        rec_state = states.get("recurrent")
        if self._recurrent_is_network:
            self.recurrent.set_states(rec_state, clone=clone, detach=detach, strict=strict)
        else:
            self.recurrent.set_state(rec_state, clone=clone, detach=detach)


class FlatPerAssetActionValueNetwork(torch.nn.Module):
    """
    Per-asset extractor → flatten → parallel (recurrent | feedforward) → heads.

    State layout (matches src.environment.generic.longshort):
        state_dim = d_global + num_assets * d_asset

    Architecture:
        input
          ├─ globals (d_global)              ──┐ broadcast across N tokens
          └─ assets  (N, d_asset)             ─┘
                                 │
                                 v
               per-asset MLP shared across N  →  (N, d_model)
                                 │
                          flatten over N
                                 v
                       (N * d_model) flat vector
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
             ┌───────────────────┴───────────────────┐
             v                                       v
          actor head                            value head
          → (action_dim) logits                → (1) value
    """

    name = "flat_per_asset_action_value"

    def __init__(
        self,
        num_assets: int,
        d_asset: int,
        d_global: int,
        action_dim: int,
        d_model: int = 64,
        hidden_dims_asset=None,
        d_mem: int = 64,
        d_ff: int = 64,
        hidden_dims_mem=None,
        hidden_dims_ff=None,
        hidden_dims_actor=None,
        hidden_dims_value=None,
        combine_mode: str = "concat",
        activation=torch.tanh,
        recurrent_activation=torch.tanh,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
    ):
        super().__init__()
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

        self.num_assets = int(num_assets)
        self.d_asset = int(d_asset)
        self.d_global = int(d_global)
        self.state_dim = self.d_global + self.num_assets * self.d_asset
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.d_mem = int(d_mem)
        self.d_ff = int(d_ff)
        self.combine_mode = combine_mode
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.recurrent_type = recurrent_type
        self.recurrent_kwargs = dict(recurrent_kwargs or {})

        self.asset_extractor = VanillaNetwork(
            self.d_asset + self.d_global,
            self.d_model,
            hidden_dims=hidden_dims_asset,
            activation=activation,
        )

        flat_dim = self.num_assets * self.d_model
        if hidden_dims_mem:
            self.recurrent = RecurrentNetwork(
                flat_dim,
                self.d_mem,
                hidden_dims=hidden_dims_mem,
                activation=recurrent_activation,
                recurrent_type=recurrent_type,
                recurrent_kwargs=self.recurrent_kwargs,
            )
        else:
            self.recurrent = _build_recurrent_cell(
                flat_dim,
                self.d_mem,
                recurrent_activation,
                recurrent_type,
                self.recurrent_kwargs,
            )
        self._recurrent_is_network = isinstance(self.recurrent, RecurrentNetwork)

        self.feedforward = VanillaNetwork(
            flat_dim,
            self.d_ff,
            hidden_dims=hidden_dims_ff,
            activation=activation,
        )

        head_in = self.d_mem if self.combine_mode == "add" else self.d_mem + self.d_ff
        self.actor = VanillaNetwork(head_in, self.action_dim, hidden_dims=hidden_dims_actor, activation=activation)
        self.value = VanillaNetwork(head_in, 1, hidden_dims=hidden_dims_value, activation=activation)

    def _encode_flat(self, x_flat: torch.Tensor) -> torch.Tensor:
        """x_flat (B, state_dim) → (B, N * d_model) — per-asset MLP then flatten."""
        B = x_flat.shape[0]
        globals_feat = x_flat[:, : self.d_global]                                          # (B, d_global)
        asset_feat = x_flat[:, self.d_global :].view(B, self.num_assets, self.d_asset)     # (B, N, d_asset)
        globals_broadcast = globals_feat.unsqueeze(1).expand(-1, self.num_assets, -1)      # (B, N, d_global)
        per_asset_in = torch.cat([asset_feat, globals_broadcast], dim=-1)                   # (B, N, d_asset+d_global)
        tokens = self.asset_extractor(per_asset_in).view(B, self.num_assets, self.d_model)  # (B, N, d_model)
        return tokens.reshape(B, -1)                                                        # (B, N * d_model)

    def _combine(self, h_mem: torch.Tensor, h_ff: torch.Tensor) -> torch.Tensor:
        if self.combine_mode == "add":
            return h_mem + h_ff
        return torch.cat([h_mem, h_ff], dim=-1)

    def reset(self, batch_size: int = 1):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.recurrent.reset(batch_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        x_flat = x.reshape(-1, self.state_dim)              # (B', state_dim)
        z = self._encode_flat(x_flat)                       # (B', N * d_model)
        h_mem = self.recurrent(z)                           # (B', d_mem)
        h_ff = self.feedforward(z)                          # (B', d_ff)
        h = self._combine(h_mem, h_ff)                      # (B', head_in)
        logits = self.actor(h).reshape(shape[:-1] + (self.action_dim,))
        values = self.value(h).squeeze(-1).reshape(shape[:-1])
        return logits, values

    def forward_seq(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x_seq.dim() == 2:
            if x_seq.shape[-1] != self.state_dim:
                raise ValueError(f"Expected last dim {self.state_dim}, got {x_seq.shape[-1]}.")
            T = x_seq.shape[0]
            B = 1
            x_batched = x_seq.unsqueeze(1)
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

        x_flat = x_batched.reshape(T * B, self.state_dim)
        z_flat = self._encode_flat(x_flat)                  # (T*B, N * d_model)
        z_seq = z_flat.view(T, B, -1)                       # (T, B, N * d_model)

        h_mem_seq = self.recurrent.forward_seq(z_seq)       # (T, B, d_mem)
        h_ff_seq = self.feedforward(z_flat).view(T, B, self.d_ff)  # (T, B, d_ff)

        h_seq = self._combine(h_mem_seq, h_ff_seq)          # (T, B, head_in)
        h_flat = h_seq.reshape(T * B, -1)

        logits_seq = self.actor(h_flat).view(T, B, self.action_dim)
        values_seq = self.value(h_flat).squeeze(-1).view(T, B)

        if squeeze_batch:
            logits_seq = logits_seq.squeeze(1)
            values_seq = values_seq.squeeze(1)
        return logits_seq, values_seq

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        if self._recurrent_is_network:
            rec_state = self.recurrent.get_states(clone=clone, detach=detach)
        else:
            rec_state = self.recurrent.get_state(clone=clone, detach=detach)
        return {"recurrent": rec_state}

    def set_states(self, states: dict, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict and "recurrent" not in states:
            raise KeyError("Missing key 'recurrent' in states snapshot.")
        rec_state = states.get("recurrent")
        if self._recurrent_is_network:
            self.recurrent.set_states(rec_state, clone=clone, detach=detach, strict=strict)
        else:
            self.recurrent.set_state(rec_state, clone=clone, detach=detach)


NETWORK_REGISTRY: dict[str, type[torch.nn.Module]] = {
    cls.name: cls
    for cls in (
        ActionValueNetwork,
        RecurrentActionValueNetwork,
        ActionQNetwork,
        RecurrentActionQNetwork,
        AttentionMemoryActionValueNetwork,
        FlatPerAssetActionValueNetwork,
    )
}


def build_network(network: dict) -> torch.nn.Module:
    """
    Build a network from a config dict.

    The dict must contain a ``type`` key whose value matches the ``name``
    attribute of a registered network class. All other keys are forwarded as
    keyword arguments to the network constructor.

    Example
    -------
    >>> build_network({
    ...     "type": "action_value",
    ...     "state_dim": 62,
    ...     "action_dim": 37,
    ...     "hidden_dims": [128, 128],
    ... })
    """
    cfg = dict(network)
    if "type" not in cfg:
        raise KeyError("network dict must contain a 'type' field.")
    type_name = cfg.pop("type")
    if type_name not in NETWORK_REGISTRY:
        raise ValueError(
            f"Unknown network type '{type_name}'. "
            f"Available: {sorted(NETWORK_REGISTRY)}."
        )
    return NETWORK_REGISTRY[type_name](**cfg)


__all__ = [
    "ActionQNetwork",
    "ActionValueNetwork",
    "FlatPerAssetActionValueNetwork",
    "NETWORK_REGISTRY",
    "AttentionMemoryActionValueNetwork",
    "RecurrentActionQNetwork",
    "RecurrentActionValueNetwork",
    "build_network",
]
