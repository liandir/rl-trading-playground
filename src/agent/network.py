import torch

from src.network.attention import ResidualSelfAttentionBlock
from src.network.utils import _build_causal_mask, _sinusoidal_pos_encoding
from src.network.vanilla import VanillaNetwork
from src.network.recurrent import RecurrentNetwork, _build_recurrent_cell


class _FeedForwardActionValueNetwork(torch.nn.Module):
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

    def reset(self, batch_size: int = 1):
        return None

    def forward_seq(self, x_seq):
        return self.forward(x_seq)

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        return {}

    def set_states(self, states: dict | None, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict and states not in ({}, None):
            raise ValueError("Feedforward ActionValueNetwork has no recurrent state.")


class ActionValueNetwork(torch.nn.Module):
    """Shared recurrent trunk with categorical-policy and scalar-value heads."""

    name = "action_value"

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


class _FeedForwardActionQNetwork(torch.nn.Module):
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
        shape = x.shape
        z = x.view(-1, self.state_dim)
        for layer in self.layers:
            z = self.activation(layer(z))
        logits = self.actor(z).view(*shape[:-1], self.action_dim)
        q_values = self.q_head(z).view(*shape[:-1], self.action_dim)
        return logits, q_values

    def reset(self, batch_size: int = 1):
        return None

    def forward_seq(self, x_seq):
        return self.forward(x_seq)

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        return {}

    def set_states(self, states: dict | None, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict and states not in ({}, None):
            raise ValueError("Feedforward ActionQNetwork has no recurrent state.")


class ActionQNetwork(torch.nn.Module):
    """Shared recurrent trunk with categorical-policy and discrete-Q heads."""

    name = "action_q"

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
        """
        x_flat : (B, state_dim)  →  tokens (B, N, token_dim)

        Splits into globals + per-asset block, runs the shared per-asset MLP,
        concatenates a learned per-asset embedding, and applies the
        transformer-style attention/FFN stack across assets. Flattening is left to
        ``_flatten_tokens`` so callers can also access the per-token
        representation.
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
        """tokens (B, N, token_dim) → (B, post_attn_dim)"""
        return tokens.reshape(tokens.shape[0], -1)

    def _build_logits(self, tokens: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, N, token_dim)
        h      : (B, d_mem + d_ff)
        returns logits (B, action_dim)
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
        if self.combine_mode == "add":
            return h_mem + h_ff
        return torch.cat([h_mem, h_ff], dim=-1)

    def _build_aux(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """h : (M, head_in) → flat aux dict with leading dim M."""
        return {
            "reward":    self.aux_reward(h).squeeze(-1),
            "asset_ret": self.aux_asset_ret(h).view(h.shape[0], self.num_assets, self.n_aux_horizons),
            "asset_vol": torch.nn.functional.softplus(
                self.aux_asset_vol(h).view(h.shape[0], self.num_assets, self.n_aux_horizons)
            ),
        }

    def reset(self, batch_size: int = 1):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.recurrent.reset(batch_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
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
        """
        x_seq : (T, state_dim) or (T, B, state_dim)

        Returns (logits, values) when aux_horizons is None, otherwise
        (logits, values, aux) — matching the AuxiliaryPerAssetActionValueNetwork
        interface so this network can be dropped in for HierarchicalAuxAACAgent.
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


class _LatentRecurrentFeedForwardBranch(torch.nn.Module):
    """Parallel recurrent/feedforward branch used by latent model heads."""

    def __init__(
        self,
        n_in: int,
        d_mem: int,
        d_ff: int,
        hidden_dims_mem=None,
        hidden_dims_ff=None,
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
        hidden_dims_mem = hidden_dims_mem or []
        hidden_dims_ff = hidden_dims_ff or []
        recurrent_kwargs = dict(recurrent_kwargs or {})

        self.n_in = int(n_in)
        self.d_mem = int(d_mem)
        self.d_ff = int(d_ff)
        self.combine_mode = combine_mode
        self.out_dim = self.d_mem if self.combine_mode == "add" else self.d_mem + self.d_ff

        if hidden_dims_mem:
            self.recurrent = RecurrentNetwork(
                self.n_in,
                self.d_mem,
                hidden_dims=hidden_dims_mem,
                activation=recurrent_activation,
                recurrent_type=recurrent_type,
                recurrent_kwargs=recurrent_kwargs,
            )
        else:
            self.recurrent = _build_recurrent_cell(
                self.n_in,
                self.d_mem,
                recurrent_activation,
                recurrent_type,
                recurrent_kwargs,
            )
        self._recurrent_is_network = isinstance(self.recurrent, RecurrentNetwork)
        self.feedforward = VanillaNetwork(
            self.n_in,
            self.d_ff,
            hidden_dims=hidden_dims_ff,
            activation=activation,
        )

    def _combine(self, h_mem: torch.Tensor, h_ff: torch.Tensor) -> torch.Tensor:
        if self.combine_mode == "add":
            return h_mem + h_ff
        return torch.cat([h_mem, h_ff], dim=-1)

    def reset(self, batch_size: int = 1, device: torch.device = None, dtype: torch.dtype = None):
        self.recurrent.reset(batch_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._combine(self.recurrent(x), self.feedforward(x))

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        shape = x_seq.shape
        h_mem = self.recurrent.forward_seq(x_seq)
        h_ff = self.feedforward(x_seq.reshape(-1, self.n_in)).view(*shape[:-1], self.d_ff)
        return self._combine(h_mem, h_ff)

    def get_state(self, clone: bool = True, detach: bool = True):
        if self._recurrent_is_network:
            return self.recurrent.get_states(clone=clone, detach=detach)
        return self.recurrent.get_state(clone=clone, detach=detach)

    def set_state(self, state, clone: bool = True, detach: bool = True, strict: bool = True):
        if self._recurrent_is_network:
            self.recurrent.set_states(state, clone=clone, detach=detach, strict=strict)
        else:
            self.recurrent.set_state(state, clone=clone, detach=detach)


class _StatelessFeedForwardBranch(torch.nn.Module):
    """Feedforward-only branch with the same small interface as recurrent branches."""

    def __init__(self, n_in: int, n_out: int | None = None, hidden_dims=None, activation=torch.tanh):
        super().__init__()
        self.n_in = int(n_in)
        self.out_dim = self.n_in if n_out is None else int(n_out)
        if n_out is None and not hidden_dims:
            self.net = torch.nn.Identity()
        else:
            self.net = VanillaNetwork(self.n_in, self.out_dim, hidden_dims=hidden_dims or [], activation=activation)

    def reset(self, batch_size: int = 1, device: torch.device = None, dtype: torch.dtype = None):
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        shape = x_seq.shape
        return self.forward(x_seq.reshape(-1, self.n_in)).view(*shape[:-1], self.out_dim)

    def get_state(self, clone: bool = True, detach: bool = True):
        return {}

    def set_state(self, state, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict and state not in ({}, None):
            raise ValueError("Stateless branch has no recurrent state.")


class AttentionMemoryModelNetwork(torch.nn.Module):
    """
    Attention-memory actor/value/model network with shared global memory.

    ``encode(x)`` returns the flattened post-attention latent ``z``. A shared
    recurrent/feedforward global memory consumes ``z``. The model head predicts
    the next latent from that global memory, so the prediction loss trains the
    shared temporal state. Actor, value, and model heads can optionally add
    their own recurrent/feedforward branch on top.
    """

    name = "attention_memory_model"

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
        actor_recurrent: bool = False,
        value_recurrent: bool = False,
        model_recurrent: bool = False,
        d_head_mem: int | None = None,
        d_head_ff: int | None = None,
        hidden_dims_actor_mem=None,
        hidden_dims_actor_ff=None,
        hidden_dims_value_mem=None,
        hidden_dims_value_ff=None,
        hidden_dims_model_mem=None,
        hidden_dims_model_ff=None,
        hidden_dims_actor=None,
        hidden_dims_value=None,
        hidden_dims_model=None,
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

        hidden_dims_asset = hidden_dims_asset or []
        hidden_dims_mem = hidden_dims_mem or []
        hidden_dims_ff = hidden_dims_ff or []
        hidden_dims_actor_mem = hidden_dims_actor_mem or []
        hidden_dims_actor_ff = hidden_dims_actor_ff or []
        hidden_dims_value_mem = hidden_dims_value_mem or []
        hidden_dims_value_ff = hidden_dims_value_ff or []
        hidden_dims_model_mem = hidden_dims_model_mem or []
        hidden_dims_model_ff = hidden_dims_model_ff or []
        hidden_dims_actor = hidden_dims_actor or []
        hidden_dims_value = hidden_dims_value or []
        hidden_dims_model = hidden_dims_model or []
        hidden_dims_per_asset_head = hidden_dims_per_asset_head or []

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
        self.d_head_mem = self.d_mem if d_head_mem is None else int(d_head_mem)
        self.d_head_ff = self.d_ff if d_head_ff is None else int(d_head_ff)
        self.combine_mode = combine_mode
        self.output_mode = output_mode
        self.n_directions = int(n_directions)
        self.actor_recurrent = bool(actor_recurrent)
        self.value_recurrent = bool(value_recurrent)
        self.model_recurrent = bool(model_recurrent)
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.recurrent_type = recurrent_type
        self.recurrent_kwargs = dict(recurrent_kwargs or {})

        if self.asset_embed_dim <= 0:
            raise ValueError(f"asset_embed_dim must be >= 1, got {self.asset_embed_dim}.")
        self.token_dim = self.d_model + self.asset_embed_dim
        if self.num_heads > 1 and self.token_dim % self.num_heads != 0:
            raise ValueError(
                f"d_model + asset_embed_dim ({self.token_dim}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )

        self.primary_action_dim = 1 + 3 * self.num_assets
        bucket_part = self.action_dim - self.primary_action_dim
        if bucket_part <= 0 or bucket_part % 2 != 0:
            raise ValueError(
                f"action_dim must match hierarchical layout 1 + 3*N + 2*K; got action_dim={self.action_dim}."
            )
        self.n_size_buckets = bucket_part // 2
        self.action_encoding_dim = self.primary_action_dim + self.n_size_buckets

        self.asset_extractor = VanillaNetwork(
            self.d_asset + self.d_global,
            self.d_model,
            hidden_dims=hidden_dims_asset,
            activation=activation,
        )
        self.asset_embedding = torch.nn.Embedding(self.num_assets, self.asset_embed_dim)
        torch.nn.init.xavier_uniform_(self.asset_embedding.weight)
        self.register_buffer("_asset_ids", torch.arange(self.num_assets, dtype=torch.long), persistent=False)

        self.attention_layers = torch.nn.ModuleList(
            ResidualSelfAttentionBlock(self.token_dim, self.num_heads, use_layer_norm=self.use_layer_norm)
            for _ in range(self.n_att_layers)
        )
        self.latent_dim = self.num_assets * self.token_dim
        self._post_attn_dim = self.latent_dim

        self.global_branch = _LatentRecurrentFeedForwardBranch(
            self.latent_dim,
            d_mem=self.d_mem,
            d_ff=self.d_ff,
            hidden_dims_mem=hidden_dims_mem,
            hidden_dims_ff=hidden_dims_ff,
            combine_mode=combine_mode,
            activation=activation,
            recurrent_activation=recurrent_activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=self.recurrent_kwargs,
        )
        self.global_dim = self.global_branch.out_dim

        self.actor_branch = self._make_head_branch(
            self.global_dim,
            self.actor_recurrent,
            hidden_dims_actor_mem,
            hidden_dims_actor_ff,
        )
        self.value_branch = self._make_head_branch(
            self.global_dim,
            self.value_recurrent,
            hidden_dims_value_mem,
            hidden_dims_value_ff,
        )
        self.model_branch = self._make_head_branch(
            self.global_dim + self.action_encoding_dim,
            self.model_recurrent,
            hidden_dims_model_mem,
            hidden_dims_model_ff,
        )
        head_in = self.actor_branch.out_dim

        self.value_head = VanillaNetwork(
            self.value_branch.out_dim,
            1,
            hidden_dims=hidden_dims_value,
            activation=activation,
        )
        self.model_head = VanillaNetwork(
            self.model_branch.out_dim,
            self.latent_dim,
            hidden_dims=hidden_dims_model,
            activation=activation,
        )

        if self.output_mode == "pooled":
            self.actor = VanillaNetwork(head_in, self.action_dim, hidden_dims=hidden_dims_actor, activation=activation)
            self.per_asset_head = None
            self.idle_head = None
        else:
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

        self._cached_global_key = None
        self._cached_global_h = None

    def _make_head_branch(
        self,
        n_in: int,
        recurrent: bool,
        hidden_dims_mem=None,
        hidden_dims_ff=None,
    ) -> torch.nn.Module:
        if recurrent:
            return _LatentRecurrentFeedForwardBranch(
                n_in,
                d_mem=self.d_head_mem,
                d_ff=self.d_head_ff,
                hidden_dims_mem=hidden_dims_mem,
                hidden_dims_ff=hidden_dims_ff,
                combine_mode=self.combine_mode,
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                recurrent_type=self.recurrent_type,
                recurrent_kwargs=self.recurrent_kwargs,
            )
        return _StatelessFeedForwardBranch(n_in)

    def _encode_tokens(self, x_flat: torch.Tensor) -> torch.Tensor:
        B = x_flat.shape[0]
        globals_feat = x_flat[:, : self.d_global]
        asset_feat = x_flat[:, self.d_global :].view(B, self.num_assets, self.d_asset)
        globals_broadcast = globals_feat.unsqueeze(1).expand(-1, self.num_assets, -1)
        per_asset_in = torch.cat([asset_feat, globals_broadcast], dim=-1)
        asset_repr = self.asset_extractor(per_asset_in).view(B, self.num_assets, self.d_model)
        asset_emb = self.asset_embedding(self._asset_ids).unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([asset_repr, asset_emb], dim=-1)
        for block in self.attention_layers:
            tokens = block(tokens)
        return tokens

    def _tokens_from_z(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1, self.num_assets, self.token_dim)

    def _build_logits(self, tokens: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "pooled":
            return self.actor(h)

        B = h.shape[0]
        h_broadcast = h.unsqueeze(1).expand(-1, self.num_assets, -1)
        per_asset_in = torch.cat([tokens, h_broadcast], dim=-1)
        per_asset_logits = self.per_asset_head(per_asset_in)
        per_asset_logits = per_asset_logits.view(B, self.num_assets, self.n_directions, self.n_size_buckets)
        parts = [self.idle_head(h)]
        for d in range(self.n_directions):
            parts.append(per_asset_logits[:, :, d, :].reshape(B, -1))
        return torch.cat(parts, dim=-1)

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        a_d = action[..., 0].long().clamp(0, self.primary_action_dim - 1)
        a_q = action[..., 1].long().clamp(0, self.n_size_buckets - 1)
        primary = torch.nn.functional.one_hot(a_d, self.primary_action_dim)
        bucket = torch.nn.functional.one_hot(a_q, self.n_size_buckets)
        return torch.cat([primary, bucket], dim=-1).to(dtype=next(self.parameters()).dtype, device=action.device)

    def _cache_key(self, z: torch.Tensor):
        if not torch.is_tensor(z):
            return None
        return (z.data_ptr(), tuple(z.shape), z._version)

    def _global(self, z: torch.Tensor, *, cache: bool = False) -> torch.Tensor:
        shape = z.shape
        z_flat = z.reshape(-1, self.latent_dim)
        h = self.global_branch(z_flat)
        if cache:
            self._cached_global_key = self._cache_key(z)
            self._cached_global_h = h
        return h.reshape(shape[:-1] + (self.global_dim,))

    def _consume_cached_global(self, z: torch.Tensor) -> torch.Tensor | None:
        key = self._cache_key(z)
        if key is None or key != self._cached_global_key:
            return None
        h = self._cached_global_h
        self._cached_global_key = None
        self._cached_global_h = None
        return h.reshape(z.shape[:-1] + (self.global_dim,))

    def _clear_cached_global(self):
        self._cached_global_key = None
        self._cached_global_h = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.state_dim)
        z = self._encode_tokens(x_flat).reshape(x_flat.shape[0], self.latent_dim)
        return z.reshape(shape[:-1] + (self.latent_dim,))

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def action(self, z: torch.Tensor, global_h: torch.Tensor | None = None) -> torch.Tensor:
        shape = z.shape
        z_flat = z.reshape(-1, self.latent_dim)
        if global_h is None:
            global_h = self._global(z)
        actor_h = self.actor_branch(global_h.reshape(-1, self.global_dim))
        logits = self._build_logits(self._tokens_from_z(z_flat), actor_h)
        return logits.reshape(shape[:-1] + (self.action_dim,))

    def value(self, z: torch.Tensor, global_h: torch.Tensor | None = None) -> torch.Tensor:
        shape = z.shape
        if global_h is None:
            global_h = self._global(z)
        value_h = self.value_branch(global_h.reshape(-1, self.global_dim))
        values = self.value_head(value_h).squeeze(-1)
        return values.reshape(shape[:-1])

    def policy_value_from_latent(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_h = self._global(z, cache=True)
        return self.action(z, global_h=global_h), self.value(z, global_h=global_h)

    def model(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        global_h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = z.shape
        if global_h is None:
            global_h = self._consume_cached_global(z)
        if global_h is None:
            global_h = self._global(z)
        action_flat = action.reshape(-1, 2).to(device=z.device)
        action_enc = self._encode_action(action_flat)
        model_in = torch.cat([global_h.reshape(-1, self.global_dim), action_enc], dim=-1)
        h = self.model_branch(model_in)
        next_z = self.model_head(h)
        return next_z.reshape(shape)

    def model_seq(self, z_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        if z_seq.shape[:-1] != action_seq.shape[:-1] or action_seq.shape[-1] != 2:
            raise ValueError(
                "Expected z_seq shape (..., latent_dim) and action_seq shape (..., 2), "
                f"got {tuple(z_seq.shape)} and {tuple(action_seq.shape)}."
            )
        global_seq = self.global_branch.forward_seq(z_seq)
        action_enc = self._encode_action(action_seq.to(device=z_seq.device))
        model_in = torch.cat([global_seq, action_enc], dim=-1)
        h = self.model_branch.forward_seq(model_in)
        return self.model_head(h.reshape(-1, self.model_branch.out_dim)).view(*z_seq.shape)

    def model_step(self, z: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        next_z = self.model(z, action)
        reward = torch.zeros(next_z.shape[:-1], dtype=next_z.dtype, device=next_z.device)
        done = torch.zeros_like(reward)
        return {"next_latent": next_z, "reward": reward, "done": done}

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.policy_value_from_latent(z)

    def forward_seq(self, x_seq: torch.Tensor):
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

        z_seq = self.encode(x_batched).view(T, B, self.latent_dim)
        global_seq = self.global_branch.forward_seq(z_seq)
        actor_h = self.actor_branch.forward_seq(global_seq)
        value_h = self.value_branch.forward_seq(global_seq)
        logits = self._build_logits(
            self._tokens_from_z(z_seq.reshape(T * B, self.latent_dim)),
            actor_h.reshape(T * B, -1),
        ).view(T, B, self.action_dim)
        values = self.value_head(value_h.reshape(T * B, -1)).squeeze(-1).view(T, B)

        if squeeze_batch:
            return logits.squeeze(1), values.squeeze(1)
        return logits, values

    def reset(self, batch_size: int = 1):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.global_branch.reset(batch_size, device=device, dtype=dtype)
        self.actor_branch.reset(batch_size, device=device, dtype=dtype)
        self.value_branch.reset(batch_size, device=device, dtype=dtype)
        self.model_branch.reset(batch_size, device=device, dtype=dtype)
        self._clear_cached_global()

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        return {
            "global": self.global_branch.get_state(clone=clone, detach=detach),
            "actor": self.actor_branch.get_state(clone=clone, detach=detach),
            "value": self.value_branch.get_state(clone=clone, detach=detach),
            "model": self.model_branch.get_state(clone=clone, detach=detach),
        }

    def set_states(self, states: dict, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict:
            for key in ("global", "actor", "value", "model"):
                if key not in states:
                    raise KeyError(f"Missing key '{key}' in states snapshot.")
        self.global_branch.set_state(states.get("global"), clone=clone, detach=detach, strict=strict)
        self.actor_branch.set_state(states.get("actor"), clone=clone, detach=detach, strict=strict)
        self.value_branch.set_state(states.get("value"), clone=clone, detach=detach, strict=strict)
        self.model_branch.set_state(states.get("model"), clone=clone, detach=detach, strict=strict)
        self._clear_cached_global()


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


class AuxiliaryPerAssetActionValueNetwork(FlatPerAssetActionValueNetwork):
    """
    ``FlatPerAssetActionValueNetwork`` plus supervised predictive heads.

    Public inference returns ``(logits, critic, aux)`` where ``critic`` is a
    scalar value by default or per-action Q-values when ``critic_type="q"``,
    and ``aux`` contains:
        reward    : (...,)
        asset_ret : (..., num_assets, aux_horizons)
        asset_vol : (..., num_assets, aux_horizons)
    """

    name = "auxiliary_per_asset_action_value"

    def __init__(
        self,
        *args,
        aux_horizons=(1, 5, 20),
        hidden_dims_aux=None,
        critic_type: str = "v",
        **kwargs,
    ):
        critic_type = str(critic_type).lower()
        if critic_type not in ("v", "q"):
            raise ValueError(f"critic_type must be 'v' or 'q', got '{critic_type}'.")
        hidden_dims_value = kwargs.get("hidden_dims_value", None)
        super().__init__(*args, **kwargs)
        hidden_dims_aux = hidden_dims_aux or []
        hidden_dims_value = hidden_dims_value or []
        self.critic_type = critic_type
        self.aux_horizons = tuple(int(h) for h in aux_horizons)
        if len(self.aux_horizons) < 1:
            raise ValueError("aux_horizons must contain at least one horizon.")
        if any(h <= 0 for h in self.aux_horizons):
            raise ValueError(f"aux_horizons must be positive, got {self.aux_horizons}.")
        self.n_aux_horizons = len(self.aux_horizons)

        head_in = self.d_mem if self.combine_mode == "add" else self.d_mem + self.d_ff
        if self.critic_type == "q":
            self.value = VanillaNetwork(
                head_in,
                self.action_dim,
                hidden_dims=hidden_dims_value,
                activation=self.activation,
            )
        self.aux_reward = VanillaNetwork(head_in, 1, hidden_dims=hidden_dims_aux, activation=self.activation)
        self.aux_asset_ret = VanillaNetwork(
            head_in,
            self.num_assets * self.n_aux_horizons,
            hidden_dims=hidden_dims_aux,
            activation=self.activation,
        )
        self.aux_asset_vol = VanillaNetwork(
            head_in,
            self.num_assets * self.n_aux_horizons,
            hidden_dims=hidden_dims_aux,
            activation=self.activation,
        )

    def _build_aux(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "reward": self.aux_reward(h).squeeze(-1),
            "asset_ret": self.aux_asset_ret(h).view(h.shape[0], self.num_assets, self.n_aux_horizons),
            "asset_vol": torch.nn.functional.softplus(
                self.aux_asset_vol(h).view(h.shape[0], self.num_assets, self.n_aux_horizons)
            ),
        }

    def _critic(self, h: torch.Tensor, prefix_shape: torch.Size) -> torch.Tensor:
        critic = self.value(h)
        if self.critic_type == "q":
            return critic.reshape(prefix_shape + (self.action_dim,))
        return critic.squeeze(-1).reshape(prefix_shape)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        shape = x.shape
        x_flat = x.reshape(-1, self.state_dim)
        z = self._encode_flat(x_flat)
        h_mem = self.recurrent(z)
        h_ff = self.feedforward(z)
        h = self._combine(h_mem, h_ff)

        logits = self.actor(h).reshape(shape[:-1] + (self.action_dim,))
        values = self._critic(h, shape[:-1])
        aux_flat = self._build_aux(h)
        aux = {
            "reward": aux_flat["reward"].reshape(shape[:-1]),
            "asset_ret": aux_flat["asset_ret"].reshape(shape[:-1] + (self.num_assets, self.n_aux_horizons)),
            "asset_vol": aux_flat["asset_vol"].reshape(shape[:-1] + (self.num_assets, self.n_aux_horizons)),
        }
        return logits, values, aux

    def forward_seq(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
        z_flat = self._encode_flat(x_flat)
        z_seq = z_flat.view(T, B, -1)

        h_mem_seq = self.recurrent.forward_seq(z_seq)
        h_ff_seq = self.feedforward(z_flat).view(T, B, self.d_ff)
        h_seq = self._combine(h_mem_seq, h_ff_seq)
        h_flat = h_seq.reshape(T * B, -1)

        logits_seq = self.actor(h_flat).view(T, B, self.action_dim)
        if self.critic_type == "q":
            values_seq = self.value(h_flat).view(T, B, self.action_dim)
        else:
            values_seq = self.value(h_flat).squeeze(-1).view(T, B)
        aux_flat = self._build_aux(h_flat)
        aux_seq = {
            "reward": aux_flat["reward"].view(T, B),
            "asset_ret": aux_flat["asset_ret"].view(T, B, self.num_assets, self.n_aux_horizons),
            "asset_vol": aux_flat["asset_vol"].view(T, B, self.num_assets, self.n_aux_horizons),
        }

        if squeeze_batch:
            logits_seq = logits_seq.squeeze(1)
            values_seq = values_seq.squeeze(1)
            aux_seq = {key: value.squeeze(1) for key, value in aux_seq.items()}
        return logits_seq, values_seq, aux_seq


class SpatiotemporalAuxiliaryPerAssetActionValueNetwork(torch.nn.Module):
    """
    Factored spatiotemporal attention with auxiliary prediction heads.

    At each decision step the network sees a sliding window of W recent
    observations and produces logits, a value estimate, and auxiliary targets.

    Pipeline:
        (B, W, state_dim)
          → shared per-asset MLP + asset embedding + sinusoidal temporal pos-enc
          → (B, W, N, token_dim)
          → causal temporal self-attention per asset   [n_temporal_layers]
          → take last timestep                          (B, N, token_dim)
          → cross-asset self-attention                  (B, N, token_dim)
          → flatten                                     (B, N·token_dim)
          → recurrent branch + feedforward branch       h
          → actor / value / aux heads

    forward(x):        single-step inference; updates a rolling buffer internally.
    forward_seq(x_seq): training on (T, B, state_dim); causal temporal attention
                        over the full rollout with window_size cap.
    """

    name = "spatiotemporal_aux_per_asset_action_value"

    def __init__(
        self,
        num_assets: int,
        d_asset: int,
        d_global: int,
        action_dim: int,
        d_model: int = 64,
        asset_embed_dim: int | None = None,
        hidden_dims_asset: list | None = None,
        window_size: int = 64,
        n_temporal_layers: int = 2,
        temporal_num_heads: int = 4,
        n_asset_layers: int = 1,
        asset_num_heads: int = 4,
        use_layer_norm: bool = True,
        attn_dropout: float = 0.0,
        d_mem: int = 512,
        d_ff: int = 512,
        hidden_dims_mem: list | None = None,
        hidden_dims_ff: list | None = None,
        hidden_dims_actor: list | None = None,
        hidden_dims_value: list | None = None,
        hidden_dims_aux: list | None = None,
        combine_mode: str = "concat",
        aux_horizons: tuple = (1, 5, 20),
        activation=torch.nn.GELU(),
        recurrent_activation=torch.tanh,
        recurrent_type: str = "gru",
        recurrent_kwargs: dict | None = None,
    ):
        super().__init__()

        if combine_mode not in ("concat", "add"):
            raise ValueError(f"combine_mode must be 'concat' or 'add', got '{combine_mode}'.")
        if combine_mode == "add" and d_mem != d_ff:
            raise ValueError(f"combine_mode='add' requires d_mem == d_ff, got {d_mem} vs {d_ff}.")

        hidden_dims_asset = list(hidden_dims_asset or [])
        hidden_dims_mem   = list(hidden_dims_mem   or [])
        hidden_dims_ff    = list(hidden_dims_ff    or [])
        hidden_dims_actor = list(hidden_dims_actor or [])
        hidden_dims_value = list(hidden_dims_value or [])
        hidden_dims_aux   = list(hidden_dims_aux   or [])

        self.num_assets       = int(num_assets)
        self.d_asset          = int(d_asset)
        self.d_global         = int(d_global)
        self.state_dim        = self.d_global + self.num_assets * self.d_asset
        self.action_dim       = int(action_dim)
        self.d_model          = int(d_model)
        self.asset_embed_dim  = int(d_model if asset_embed_dim is None else asset_embed_dim)
        self.token_dim        = self.d_model + self.asset_embed_dim
        self.window_size      = int(window_size)
        self.d_mem            = int(d_mem)
        self.d_ff             = int(d_ff)
        self.combine_mode     = combine_mode
        self.activation       = activation
        self.recurrent_type   = recurrent_type
        self.recurrent_kwargs = dict(recurrent_kwargs or {})
        self.aux_horizons     = tuple(int(h) for h in aux_horizons)
        self.n_aux_horizons   = len(self.aux_horizons)
        self._post_attn_dim   = self.num_assets * self.token_dim

        if self.token_dim % temporal_num_heads != 0:
            raise ValueError(
                f"token_dim={self.token_dim} must be divisible by temporal_num_heads={temporal_num_heads}."
            )
        if self.token_dim % asset_num_heads != 0:
            raise ValueError(
                f"token_dim={self.token_dim} must be divisible by asset_num_heads={asset_num_heads}."
            )

        # per-asset tokenizer
        self.asset_extractor = VanillaNetwork(
            self.d_asset + self.d_global, self.d_model,
            hidden_dims=hidden_dims_asset, activation=activation,
        )
        self.asset_embedding = torch.nn.Embedding(self.num_assets, self.asset_embed_dim)
        torch.nn.init.xavier_uniform_(self.asset_embedding.weight)
        self.register_buffer("_asset_ids", torch.arange(self.num_assets, dtype=torch.long), persistent=False)

        # causal temporal self-attention (per asset independently)
        self.temporal_layers = torch.nn.ModuleList([
            ResidualSelfAttentionBlock(
                self.token_dim, temporal_num_heads,
                use_layer_norm=use_layer_norm, attn_dropout=attn_dropout,
            )
            for _ in range(n_temporal_layers)
        ])

        # cross-asset self-attention (at the current timestep)
        self.asset_layers = torch.nn.ModuleList([
            ResidualSelfAttentionBlock(
                self.token_dim, asset_num_heads,
                use_layer_norm=use_layer_norm, attn_dropout=attn_dropout,
            )
            for _ in range(n_asset_layers)
        ])

        # recurrent + feedforward trunk (beyond-window memory)
        if hidden_dims_mem:
            self.recurrent = RecurrentNetwork(
                self._post_attn_dim, self.d_mem,
                hidden_dims=hidden_dims_mem, activation=recurrent_activation,
                recurrent_type=recurrent_type, recurrent_kwargs=self.recurrent_kwargs,
            )
        else:
            self.recurrent = _build_recurrent_cell(
                self._post_attn_dim, self.d_mem,
                recurrent_activation, recurrent_type, self.recurrent_kwargs,
            )
        self._recurrent_is_network = isinstance(self.recurrent, RecurrentNetwork)

        self.feedforward = VanillaNetwork(
            self._post_attn_dim, self.d_ff,
            hidden_dims=hidden_dims_ff, activation=activation,
        )

        head_in = self.d_mem if combine_mode == "add" else self.d_mem + self.d_ff
        self.actor         = VanillaNetwork(head_in, self.action_dim,
                                            hidden_dims=hidden_dims_actor, activation=activation)
        self.value         = VanillaNetwork(head_in, 1,
                                            hidden_dims=hidden_dims_value, activation=activation)
        self.aux_reward    = VanillaNetwork(head_in, 1,
                                            hidden_dims=hidden_dims_aux, activation=activation)
        self.aux_asset_ret = VanillaNetwork(head_in, self.num_assets * self.n_aux_horizons,
                                            hidden_dims=hidden_dims_aux, activation=activation)
        self.aux_asset_vol = VanillaNetwork(head_in, self.num_assets * self.n_aux_horizons,
                                            hidden_dims=hidden_dims_aux, activation=activation)

        self._window_buffer: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _tokenize(self, x_flat: torch.Tensor) -> torch.Tensor:
        """x_flat (M, state_dim) → tokens (M, N, token_dim)."""
        M, N = x_flat.shape[0], self.num_assets
        g    = x_flat[:, :self.d_global]                                         # (M, d_global)
        a    = x_flat[:, self.d_global:].reshape(M, N, self.d_asset)             # (M, N, d_asset)
        g_bc = g.unsqueeze(1).expand(-1, N, -1)                                  # (M, N, d_global)
        r    = self.asset_extractor(torch.cat([a, g_bc], dim=-1))                # (M*N, d_model)
        r    = r.reshape(M, N, self.d_model)
        e    = self.asset_embedding(self._asset_ids).unsqueeze(0).expand(M, -1, -1)  # (M, N, emb)
        return torch.cat([r, e], dim=-1)                                          # (M, N, token_dim)

    def _temporal_attn(self, u: torch.Tensor) -> torch.Tensor:
        """u (B, T, N, D) → v (B, T, N, D): causal per-asset temporal attention."""
        B, T, N, D = u.shape
        mask  = _build_causal_mask(T, self.window_size, u.device)
        u_bn  = u.permute(0, 2, 1, 3).reshape(B * N, T, D)   # (B*N, T, D)
        for layer in self.temporal_layers:
            u_bn = layer(u_bn, mask=mask)
        return u_bn.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)

    def _asset_attn(self, v: torch.Tensor) -> torch.Tensor:
        """v (M, N, D) → c (M, N, D): unconstrained cross-asset attention."""
        for layer in self.asset_layers:
            v = layer(v)
        return v

    def _combine(self, h_mem: torch.Tensor, h_ff: torch.Tensor) -> torch.Tensor:
        return h_mem + h_ff if self.combine_mode == "add" else torch.cat([h_mem, h_ff], dim=-1)

    def _build_aux(self, h: torch.Tensor, prefix: torch.Size) -> dict:
        M = h.shape[0]
        return {
            "reward":    self.aux_reward(h).squeeze(-1).reshape(prefix),
            "asset_ret": self.aux_asset_ret(h)
                             .reshape(M, self.num_assets, self.n_aux_horizons)
                             .reshape(prefix + (self.num_assets, self.n_aux_horizons)),
            "asset_vol": torch.nn.functional.softplus(
                             self.aux_asset_vol(h)
                             .reshape(M, self.num_assets, self.n_aux_horizons)
                         ).reshape(prefix + (self.num_assets, self.n_aux_horizons)),
        }

    # ------------------------------------------------------------------
    # state management
    # ------------------------------------------------------------------

    def reset(self, batch_size: int = 1):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        self._window_buffer = torch.zeros(
            batch_size, self.window_size, self.state_dim, device=device, dtype=dtype,
        )
        self.recurrent.reset(batch_size, device=device, dtype=dtype)

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        if self._recurrent_is_network:
            rec = self.recurrent.get_states(clone=clone, detach=detach)
        else:
            rec = self.recurrent.get_state(clone=clone, detach=detach)
        buf = self._window_buffer
        if buf is not None:
            if detach: buf = buf.detach()
            if clone:  buf = buf.clone()
        return {"recurrent": rec, "buffer": buf}

    def set_states(self, states: dict, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict and "recurrent" not in states:
            raise KeyError("Missing key 'recurrent' in states snapshot.")
        rec_state = states.get("recurrent")
        if self._recurrent_is_network:
            self.recurrent.set_states(rec_state, clone=clone, detach=detach, strict=strict)
        else:
            self.recurrent.set_state(rec_state, clone=clone, detach=detach)
        buf = states.get("buffer")
        if buf is not None:
            self._window_buffer = buf.clone() if clone else buf

    # ------------------------------------------------------------------
    # forward: single-step inference with rolling buffer
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x : (state_dim,) or (B, state_dim)

        Shifts the internal W-step buffer, appends x, runs the full
        spatiotemporal pipeline and returns (logits, values, aux).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]
        if self._window_buffer is None or self._window_buffer.shape[0] != B:
            self.reset(B)

        self._window_buffer = torch.cat(
            [self._window_buffer[:, 1:, :], x.unsqueeze(1)], dim=1
        )  # (B, W, state_dim)

        W      = self.window_size
        x_flat = self._window_buffer.reshape(B * W, self.state_dim)
        tokens = self._tokenize(x_flat)                                   # (B*W, N, D)
        u      = tokens.reshape(B, W, self.num_assets, self.token_dim)    # (B, W, N, D)
        enc    = _sinusoidal_pos_encoding(W, self.token_dim, u.device, u.dtype)
        u      = u + enc[None, :, None, :]                                # (B, W, N, D)

        v     = self._temporal_attn(u)                                    # (B, W, N, D)
        v_now = v[:, -1, :, :]                                            # (B, N, D)
        c     = self._asset_attn(v_now)                                   # (B, N, D)
        z     = c.reshape(B, -1)                                          # (B, post_attn_dim)

        h_mem = self.recurrent(z)                                         # (B, d_mem)
        h_ff  = self.feedforward(z)                                       # (B, d_ff)
        h     = self._combine(h_mem, h_ff)                                # (B, head_in)

        logits = self.actor(h)                                            # (B, action_dim)
        values = self.value(h).squeeze(-1)                                # (B,)
        aux    = self._build_aux(h, x.shape[:-1])
        return logits, values, aux

    # ------------------------------------------------------------------
    # forward_seq: training over a full rollout sequence
    # ------------------------------------------------------------------

    def forward_seq(self, x_seq: torch.Tensor) -> tuple:
        """
        x_seq : (T, state_dim) or (T, B, state_dim)

        Causal temporal attention runs over the full T dimension with a
        window_size cap, so position t attends to [max(0, t-W+1) .. t].
        Cross-asset attention and the recurrent+FF trunk run at every t.
        """
        if x_seq.dim() == 2:
            T, _  = x_seq.shape
            B, sq = 1, True
            x_b   = x_seq.unsqueeze(1)                   # (T, 1, state_dim)
        elif x_seq.dim() == 3:
            T, B, _ = x_seq.shape
            sq      = False
            x_b     = x_seq
        else:
            raise ValueError(f"Expected (T, S) or (T, B, S), got {tuple(x_seq.shape)}.")

        # tokenise all T*B frames in one batched call
        x_flat   = x_b.reshape(T * B, self.state_dim)
        tok_flat = self._tokenize(x_flat)                                 # (T*B, N, D)
        u        = tok_flat.reshape(T, B, self.num_assets, self.token_dim)
        u        = u.permute(1, 0, 2, 3)                                  # (B, T, N, D)

        enc = _sinusoidal_pos_encoding(T, self.token_dim, u.device, u.dtype)
        u   = u + enc[None, :, None, :]                                   # (B, T, N, D)

        v   = self._temporal_attn(u)                                      # (B, T, N, D)
        v   = v.permute(1, 0, 2, 3)                                       # (T, B, N, D)

        # cross-asset attention at every step
        v_flat = v.reshape(T * B, self.num_assets, self.token_dim)
        c_flat = self._asset_attn(v_flat)                                 # (T*B, N, D)
        z_flat = c_flat.reshape(T * B, -1)                                # (T*B, post_attn_dim)
        z_seq  = z_flat.reshape(T, B, -1)                                 # (T, B, post_attn_dim)

        h_mem_seq = self.recurrent.forward_seq(z_seq)                     # (T, B, d_mem)
        h_ff_seq  = self.feedforward(z_flat).reshape(T, B, self.d_ff)    # (T, B, d_ff)
        h_seq     = self._combine(h_mem_seq, h_ff_seq)                    # (T, B, head_in)
        h_flat    = h_seq.reshape(T * B, -1)

        logits_seq = self.actor(h_flat).reshape(T, B, self.action_dim)
        values_seq = self.value(h_flat).squeeze(-1).reshape(T, B)
        aux_seq    = {
            "reward":    self.aux_reward(h_flat).squeeze(-1).reshape(T, B),
            "asset_ret": self.aux_asset_ret(h_flat)
                             .reshape(T * B, self.num_assets, self.n_aux_horizons)
                             .reshape(T, B, self.num_assets, self.n_aux_horizons),
            "asset_vol": torch.nn.functional.softplus(
                             self.aux_asset_vol(h_flat)
                             .reshape(T * B, self.num_assets, self.n_aux_horizons)
                         ).reshape(T, B, self.num_assets, self.n_aux_horizons),
        }

        if sq:
            logits_seq = logits_seq.squeeze(1)
            values_seq = values_seq.squeeze(1)
            aux_seq    = {k: v.squeeze(1) for k, v in aux_seq.items()}

        return logits_seq, values_seq, aux_seq


NETWORK_REGISTRY: dict[str, type[torch.nn.Module]] = {
    cls.name: cls
    for cls in (
        ActionValueNetwork,
        ActionQNetwork,
        AttentionMemoryActionValueNetwork,
        AttentionMemoryModelNetwork,
        FlatPerAssetActionValueNetwork,
        AuxiliaryPerAssetActionValueNetwork,
        SpatiotemporalAuxiliaryPerAssetActionValueNetwork,
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
    "AttentionMemoryModelNetwork",
    "AuxiliaryPerAssetActionValueNetwork",
    "FlatPerAssetActionValueNetwork",
    "NETWORK_REGISTRY",
    "AttentionMemoryActionValueNetwork",
    "SpatiotemporalAuxiliaryPerAssetActionValueNetwork",
    "build_network",
]
