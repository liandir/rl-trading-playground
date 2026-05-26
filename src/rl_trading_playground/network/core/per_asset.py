"""Per asset utilities for neural network architectures and reusable model components."""
from collections.abc import Callable
import torch

from rl_trading_playground.network.core.recurrent import RecurrentNetwork, _build_recurrent_cell
from rl_trading_playground.network.core.vanilla import VanillaNetwork


class BaseFlatPerAssetActionValueNetwork(torch.nn.Module):
    """
    Per-asset extractor → flatten → parallel (recurrent | feedforward) → heads.

    For asset features $a_i$ and global features $g$, each asset token is
    encoded by a shared extractor
    \\[
        e_i = E_\\theta([a_i, g]), \\qquad i=1,\\dots,N.
    \\]
    The flattened token vector $e=[e_1,\\dots,e_N]$ is sent through recurrent
    and feedforward branches, then the combined representation $h$ parameterizes
    actor logits and value:
    \\[
        \\ell = A_\\psi(h), \\qquad V = C_\\omega(h).
    \\]

    State layout (matches rl_trading_playground.environment.generic.longshort):
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
            hidden_dims_asset (Any): The hidden dims asset value. Defaults to ``None``.
            d_mem (int): The d mem value. Defaults to ``64``.
            d_ff (int): The d ff value. Defaults to ``64``.
            hidden_dims_mem (Any): The hidden dims mem value. Defaults to ``None``.
            hidden_dims_ff (Any): The hidden dims ff value. Defaults to ``None``.
            hidden_dims_actor (Any): The hidden dims actor value. Defaults to ``None``.
            hidden_dims_value (Any): The hidden dims value value. Defaults to ``None``.
            combine_mode (str): The combine mode value. Defaults to ``'concat'``.
            activation (Callable[[torch.Tensor], torch.Tensor]): The activation value. Defaults to ``torch.tanh``.
            recurrent_activation (Callable[[torch.Tensor], torch.Tensor]): The recurrent activation value. Defaults to ``torch.tanh``.
            recurrent_type (str): The recurrent type value. Defaults to ``'simple'``.
            recurrent_kwargs (dict | None): The recurrent kwargs value. Defaults to ``None``.

        Returns:
            None: This function does not return a value.
        """
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
        """x_flat (B, state_dim) → (B, N * d_model) — per-asset MLP then flatten.

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
        tokens = self.asset_extractor(per_asset_in).view(B, self.num_assets, self.d_model)  # (B, N, d_model)
        return tokens.reshape(B, -1)                                                        # (B, N * d_model)

    def _combine(self, h_mem: torch.Tensor, h_ff: torch.Tensor) -> torch.Tensor:
        """Combine for BaseFlatPerAssetActionValueNetwork.

        Args:
            h_mem (torch.Tensor): The h mem value.
            h_ff (torch.Tensor): The h ff value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        if self.combine_mode == "add":
            return h_mem + h_ff
        return torch.cat([h_mem, h_ff], dim=-1)

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the forward pass.

        Args:
            x (torch.Tensor): The x value.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The computed or requested result.
        """
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
        """Compute a forward pass over a sequence.

        Args:
            x_seq (torch.Tensor): The x seq value.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The computed or requested result.
        """
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
