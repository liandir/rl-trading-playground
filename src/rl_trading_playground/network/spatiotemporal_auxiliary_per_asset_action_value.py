"""Spatiotemporal auxiliary per asset action value utilities for neural network architectures and reusable model components."""
from collections.abc import Callable
import torch

from rl_trading_playground.network.core.attention import ResidualSelfAttentionBlock
from rl_trading_playground.network.core.recurrent import RecurrentNetwork, _build_recurrent_cell
from rl_trading_playground.network.core.utils import _build_causal_mask, _sinusoidal_pos_encoding
from rl_trading_playground.network.core.vanilla import VanillaNetwork


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
        recurrent_activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        recurrent_type: str = "gru",
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
            hidden_dims_asset (list | None): The hidden dims asset value. Defaults to ``None``.
            window_size (int): The window size value. Defaults to ``64``.
            n_temporal_layers (int): The n temporal layers value. Defaults to ``2``.
            temporal_num_heads (int): The temporal num heads value. Defaults to ``4``.
            n_asset_layers (int): The n asset layers value. Defaults to ``1``.
            asset_num_heads (int): The asset num heads value. Defaults to ``4``.
            use_layer_norm (bool): The use layer norm value. Defaults to ``True``.
            attn_dropout (float): The attn dropout value. Defaults to ``0.0``.
            d_mem (int): The d mem value. Defaults to ``512``.
            d_ff (int): The d ff value. Defaults to ``512``.
            hidden_dims_mem (list | None): The hidden dims mem value. Defaults to ``None``.
            hidden_dims_ff (list | None): The hidden dims ff value. Defaults to ``None``.
            hidden_dims_actor (list | None): The hidden dims actor value. Defaults to ``None``.
            hidden_dims_value (list | None): The hidden dims value value. Defaults to ``None``.
            hidden_dims_aux (list | None): The hidden dims aux value. Defaults to ``None``.
            combine_mode (str): The combine mode value. Defaults to ``'concat'``.
            aux_horizons (tuple): The aux horizons value. Defaults to ``(1, 5, 20)``.
            activation (Any): The activation value. Defaults to ``torch.nn.GELU()``.
            recurrent_activation (Callable[[torch.Tensor], torch.Tensor]): The recurrent activation value. Defaults to ``torch.tanh``.
            recurrent_type (str): The recurrent type value. Defaults to ``'gru'``.
            recurrent_kwargs (dict | None): The recurrent kwargs value. Defaults to ``None``.

        Returns:
            None: This function does not return a value.
        """
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
        """x_flat (M, state_dim) → tokens (M, N, token_dim).

        Args:
            x_flat (torch.Tensor): The x flat value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        M, N = x_flat.shape[0], self.num_assets
        g    = x_flat[:, :self.d_global]                                         # (M, d_global)
        a    = x_flat[:, self.d_global:].reshape(M, N, self.d_asset)             # (M, N, d_asset)
        g_bc = g.unsqueeze(1).expand(-1, N, -1)                                  # (M, N, d_global)
        r    = self.asset_extractor(torch.cat([a, g_bc], dim=-1))                # (M*N, d_model)
        r    = r.reshape(M, N, self.d_model)
        e    = self.asset_embedding(self._asset_ids).unsqueeze(0).expand(M, -1, -1)  # (M, N, emb)
        return torch.cat([r, e], dim=-1)                                          # (M, N, token_dim)

    def _temporal_attn(self, u: torch.Tensor) -> torch.Tensor:
        """u (B, T, N, D) → v (B, T, N, D): causal per-asset temporal attention.

        Args:
            u (torch.Tensor): The u value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        B, T, N, D = u.shape
        mask  = _build_causal_mask(T, self.window_size, u.device)
        u_bn  = u.permute(0, 2, 1, 3).reshape(B * N, T, D)   # (B*N, T, D)
        for layer in self.temporal_layers:
            u_bn = layer(u_bn, mask=mask)
        return u_bn.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)

    def _asset_attn(self, v: torch.Tensor) -> torch.Tensor:
        """v (M, N, D) → c (M, N, D): unconstrained cross-asset attention.

        Args:
            v (torch.Tensor): The v value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        for layer in self.asset_layers:
            v = layer(v)
        return v

    def _combine(self, h_mem: torch.Tensor, h_ff: torch.Tensor) -> torch.Tensor:
        """Combine for SpatiotemporalAuxiliaryPerAssetActionValueNetwork.

        Args:
            h_mem (torch.Tensor): The h mem value.
            h_ff (torch.Tensor): The h ff value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return h_mem + h_ff if self.combine_mode == "add" else torch.cat([h_mem, h_ff], dim=-1)

    def _build_aux(self, h: torch.Tensor, prefix: torch.Size) -> dict:
        """Build the aux.

        Args:
            h (torch.Tensor): The h value.
            prefix (torch.Size): The prefix value.

        Returns:
            dict: The computed or requested result.
        """
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

    def reset(self, batch_size: int = 1) -> None:
        """Reset internal state for a new episode or stream.

        Args:
            batch_size (int): The batch size value. Defaults to ``1``.

        Returns:
            None: This function does not return a value.
        """
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        self._window_buffer = torch.zeros(
            batch_size, self.window_size, self.state_dim, device=device, dtype=dtype,
        )
        self.recurrent.reset(batch_size, device=device, dtype=dtype)

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        """Return a snapshot of recurrent states.

        Args:
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.

        Returns:
            dict: The computed or requested result.
        """
        if self._recurrent_is_network:
            rec = self.recurrent.get_states(clone=clone, detach=detach)
        else:
            rec = self.recurrent.get_state(clone=clone, detach=detach)
        buf = self._window_buffer
        if buf is not None:
            if detach: buf = buf.detach()
            if clone:  buf = buf.clone()
        return {"recurrent": rec, "buffer": buf}

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
        buf = states.get("buffer")
        if buf is not None:
            self._window_buffer = buf.clone() if clone else buf

    # ------------------------------------------------------------------
    # forward: single-step inference with rolling buffer
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple:
        """x : (state_dim,) or (B, state_dim)

        Shifts the internal W-step buffer, appends x, runs the full
        spatiotemporal pipeline and returns (logits, values, aux).

        Args:
            x (torch.Tensor): The x value.

        Returns:
            tuple: The computed or requested result.
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
        """x_seq : (T, state_dim) or (T, B, state_dim)

        Causal temporal attention runs over the full T dimension with a
        window_size cap, so position t attends to [max(0, t-W+1) .. t].
        Cross-asset attention and the recurrent+FF trunk run at every t.

        Args:
            x_seq (torch.Tensor): The x seq value.

        Returns:
            tuple: The computed or requested result.
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
