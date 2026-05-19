"""Attention memory model utilities for neural network architectures and reusable model components."""
from collections.abc import Callable
import torch

from src.network.core.attention import ResidualSelfAttentionBlock
from src.network.core.branches import LatentRecurrentFeedForwardBranch
from src.network.core.recurrent import RecurrentNetwork
from src.network.core.vanilla import VanillaNetwork


class AttentionMemoryModelNetwork(torch.nn.Module):
    """
    Attention-memory actor/value/model network with shared global memory.

    ``encode(x)`` returns the flattened post-attention latent ``z``. A shared
    parallel recurrent + feedforward global memory consumes ``z`` and produces
    ``global_h``. Each of the actor, value, and model heads is a single network
    on top of ``global_h``: a ``RecurrentNetwork`` when its ``*_recurrent`` flag
    is True, otherwise a ``VanillaNetwork``.
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
        combine_mode: str = "concat",
        actor_recurrent: bool = False,
        value_recurrent: bool = False,
        model_recurrent: bool = False,
        hidden_dims_actor=None,
        hidden_dims_value=None,
        hidden_dims_model=None,
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
            combine_mode (str): The combine mode value. Defaults to ``'concat'``.
            actor_recurrent (bool): The actor recurrent value. Defaults to ``False``.
            value_recurrent (bool): The value recurrent value. Defaults to ``False``.
            model_recurrent (bool): The model recurrent value. Defaults to ``False``.
            hidden_dims_actor (Any): The hidden dims actor value. Defaults to ``None``.
            hidden_dims_value (Any): The hidden dims value value. Defaults to ``None``.
            hidden_dims_model (Any): The hidden dims model value. Defaults to ``None``.
            activation (Callable[[torch.Tensor], torch.Tensor]): The activation value. Defaults to ``torch.tanh``.
            recurrent_activation (Callable[[torch.Tensor], torch.Tensor]): The recurrent activation value. Defaults to ``torch.tanh``.
            recurrent_type (str): The recurrent type value. Defaults to ``'simple'``.
            recurrent_kwargs (dict | None): The recurrent kwargs value. Defaults to ``None``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__()
        hidden_dims_asset = hidden_dims_asset or []
        hidden_dims_mem = hidden_dims_mem or []
        hidden_dims_ff = hidden_dims_ff or []
        hidden_dims_actor = hidden_dims_actor or []
        hidden_dims_value = hidden_dims_value or []
        hidden_dims_model = hidden_dims_model or []

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

        self.global_branch = LatentRecurrentFeedForwardBranch(
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

        self.actor = self._build_head(self.global_dim, self.action_dim, self.actor_recurrent, hidden_dims_actor)
        self.value_head = self._build_head(self.global_dim, 1, self.value_recurrent, hidden_dims_value)
        self.model_head = self._build_head(
            self.global_dim + self.action_encoding_dim, self.latent_dim, self.model_recurrent, hidden_dims_model,
        )

    def _build_head(self, n_in: int, n_out: int, recurrent: bool, hidden_dims: list[int]) -> torch.nn.Module:
        """Build the head.

        Args:
            n_in (int): The n in value.
            n_out (int): The n out value.
            recurrent (bool): The recurrent value.
            hidden_dims (list[int]): The hidden dims value.

        Returns:
            torch.nn.Module: The computed or requested result.
        """
        if recurrent:
            return RecurrentNetwork(
                n_in,
                n_out,
                hidden_dims=hidden_dims,
                activation=self.recurrent_activation,
                recurrent_type=self.recurrent_type,
                recurrent_kwargs=self.recurrent_kwargs,
            )
        return VanillaNetwork(n_in, n_out, hidden_dims=hidden_dims, activation=self.activation)

    def _encode_tokens(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Encode tokens for AttentionMemoryModelNetwork.

        Args:
            x_flat (torch.Tensor): The x flat value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
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

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """Encode action for AttentionMemoryModelNetwork.

        Args:
            action (torch.Tensor): The action value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        a_d = action[..., 0].long().clamp(0, self.primary_action_dim - 1)
        a_q = action[..., 1].long().clamp(0, self.n_size_buckets - 1)
        primary = torch.nn.functional.one_hot(a_d, self.primary_action_dim)
        bucket = torch.nn.functional.one_hot(a_q, self.n_size_buckets)
        return torch.cat([primary, bucket], dim=-1).to(dtype=next(self.parameters()).dtype, device=action.device)

    def _global(self, z: torch.Tensor) -> torch.Tensor:
        """Global for AttentionMemoryModelNetwork.

        Args:
            z (torch.Tensor): The z value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        shape = z.shape
        h = self.global_branch(z.reshape(-1, self.latent_dim))
        return h.reshape(shape[:-1] + (self.global_dim,))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode for AttentionMemoryModelNetwork.

        Args:
            x (torch.Tensor): The x value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.state_dim)
        z = self._encode_tokens(x_flat).reshape(x_flat.shape[0], self.latent_dim)
        return z.reshape(shape[:-1] + (self.latent_dim,))

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode latent for AttentionMemoryModelNetwork.

        Args:
            x (torch.Tensor): The x value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return self.encode(x)

    def action(self, z: torch.Tensor, global_h: torch.Tensor | None = None) -> torch.Tensor:
        """Action for AttentionMemoryModelNetwork.

        Args:
            z (torch.Tensor): The z value.
            global_h (torch.Tensor | None): The global h value. Defaults to ``None``.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        if global_h is None:
            global_h = self._global(z)
        shape = z.shape
        logits = self.actor(global_h.reshape(-1, self.global_dim))
        return logits.reshape(shape[:-1] + (self.action_dim,))

    def value(self, z: torch.Tensor, global_h: torch.Tensor | None = None) -> torch.Tensor:
        """Value for AttentionMemoryModelNetwork.

        Args:
            z (torch.Tensor): The z value.
            global_h (torch.Tensor | None): The global h value. Defaults to ``None``.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        if global_h is None:
            global_h = self._global(z)
        shape = z.shape
        return self.value_head(global_h.reshape(-1, self.global_dim)).squeeze(-1).reshape(shape[:-1])

    def policy_value_from_latent(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy value from latent for AttentionMemoryModelNetwork.

        Args:
            z (torch.Tensor): The z value.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The computed or requested result.
        """
        global_h = self._global(z)
        return self.action(z, global_h=global_h), self.value(z, global_h=global_h)

    def model(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        global_h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Model for AttentionMemoryModelNetwork.

        Args:
            z (torch.Tensor): The z value.
            action (torch.Tensor): The action value.
            global_h (torch.Tensor | None): The global h value. Defaults to ``None``.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        if global_h is None:
            global_h = self._global(z)
        action_enc = self._encode_action(action.reshape(-1, 2).to(device=z.device))
        model_in = torch.cat([global_h.reshape(-1, self.global_dim), action_enc], dim=-1)
        return self.model_head(model_in).reshape(z.shape)

    def model_seq(self, z_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """Model seq for AttentionMemoryModelNetwork.

        Args:
            z_seq (torch.Tensor): The z seq value.
            action_seq (torch.Tensor): The action seq value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        if z_seq.shape[:-1] != action_seq.shape[:-1] or action_seq.shape[-1] != 2:
            raise ValueError(
                "Expected z_seq shape (..., latent_dim) and action_seq shape (..., 2), "
                f"got {tuple(z_seq.shape)} and {tuple(action_seq.shape)}."
            )
        global_seq = self.global_branch.forward_seq(z_seq)
        action_enc = self._encode_action(action_seq.to(device=z_seq.device))
        return self.model_head.forward_seq(torch.cat([global_seq, action_enc], dim=-1))

    def model_step(self, z: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Model step for AttentionMemoryModelNetwork.

        Args:
            z (torch.Tensor): The z value.
            action (torch.Tensor): The action value.

        Returns:
            dict[str, torch.Tensor]: The computed or requested result.
        """
        next_z = self.model(z, action)
        reward = torch.zeros(next_z.shape[:-1], dtype=next_z.dtype, device=next_z.device)
        done = torch.zeros_like(reward)
        return {"next_latent": next_z, "reward": reward, "done": done}

    def forward(self, x: torch.Tensor):
        """Compute the forward pass.

        Args:
            x (torch.Tensor): The x value.

        Returns:
            Any: The computed or requested result.
        """
        return self.policy_value_from_latent(self.encode(x))

    def forward_seq(self, x_seq: torch.Tensor):
        """Compute a forward pass over a sequence.

        Args:
            x_seq (torch.Tensor): The x seq value.

        Returns:
            Any: The computed or requested result.
        """
        if x_seq.dim() not in (2, 3) or x_seq.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected sequence input with shape (T, {self.state_dim}) or "
                f"(T, B, {self.state_dim}), got {tuple(x_seq.shape)}."
            )
        squeeze_batch = x_seq.dim() == 2
        x_batched = x_seq.unsqueeze(1) if squeeze_batch else x_seq
        T, B = x_batched.shape[:2]

        z_seq = self.encode(x_batched).view(T, B, self.latent_dim)
        global_seq = self.global_branch.forward_seq(z_seq)
        logits = self.actor.forward_seq(global_seq)
        values = self.value_head.forward_seq(global_seq).squeeze(-1)

        if squeeze_batch:
            return logits.squeeze(1), values.squeeze(1)
        return logits, values

    def reset(self, batch_size: int = 1) -> None:
        """Reset internal state for a new episode or stream.

        Args:
            batch_size (int): The batch size value. Defaults to ``1``.

        Returns:
            None: This function does not return a value.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.global_branch.reset(batch_size, device=device, dtype=dtype)
        self.actor.reset(batch_size, device=device, dtype=dtype)
        self.value_head.reset(batch_size, device=device, dtype=dtype)
        self.model_head.reset(batch_size, device=device, dtype=dtype)

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        """Return a snapshot of recurrent states.

        Args:
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.

        Returns:
            dict: The computed or requested result.
        """
        return {
            "global": self.global_branch.get_state(clone=clone, detach=detach),
            "actor": self.actor.get_states(clone=clone, detach=detach),
            "value": self.value_head.get_states(clone=clone, detach=detach),
            "model": self.model_head.get_states(clone=clone, detach=detach),
        }

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
        if strict:
            for key in ("global", "actor", "value", "model"):
                if key not in states:
                    raise KeyError(f"Missing key '{key}' in states snapshot.")
        self.global_branch.set_state(states.get("global"), clone=clone, detach=detach, strict=strict)
        self.actor.set_states(states.get("actor", {}), clone=clone, detach=detach, strict=strict)
        self.value_head.set_states(states.get("value", {}), clone=clone, detach=detach, strict=strict)
        self.model_head.set_states(states.get("model", {}), clone=clone, detach=detach, strict=strict)
