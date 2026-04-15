import torch

from src.network.vanilla import VanillaNetwork
from src.network.recurrent import RecurrentNetwork, _build_recurrent_cell


BETA_EPS = 1e-3


def _split_alpha_beta(ab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ab = torch.nn.functional.softplus(ab) + BETA_EPS
    alpha = ab[..., 0]
    beta = ab[..., 1]
    return alpha, beta


class HybridActionValueNetwork(torch.nn.Module):
    """Shared feedforward trunk with joint hybrid-policy and scalar-value heads."""

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
        self.actor_out_dim = action_dim + 2
        self.activation = activation

        self.layers = torch.nn.ModuleList()
        prev = state_dim
        for h in hidden_dims:
            self.layers.append(torch.nn.Linear(prev, h))
            prev = h

        self.actor = VanillaNetwork(prev, self.actor_out_dim, hidden_dims=hidden_dims_actor, activation=activation)
        self.v_head = VanillaNetwork(prev, 1, hidden_dims=hidden_dims_value, activation=activation)

    def forward(self, x):
        shape = x.shape  # (..., state_dim)
        z = x.reshape(-1, self.state_dim)
        for layer in self.layers:
            z = self.activation(layer(z))
        actor_out = self.actor(z).view(*shape[:-1], self.actor_out_dim)
        logits = actor_out[..., :self.action_dim]
        ab = actor_out[..., self.action_dim:]
        alpha, beta = _split_alpha_beta(ab)
        values = self.v_head(z).squeeze(-1).view(*shape[:-1])
        return logits, alpha, beta, values


class RecurrentHybridActionValueNetwork(torch.nn.Module):
    """Shared recurrent trunk with joint hybrid-policy and scalar-value heads."""

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
        self.actor_out_dim = action_dim + 2
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
            self.actor_out_dim,
            hidden_dims=hidden_dims_actor,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=self.recurrent_kwargs,
        )
        self.v_head = RecurrentNetwork(
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
        self.v_head.reset(batch_size, device=device, dtype=dtype)

    def forward(self, x):
        z = x.reshape(-1, self.state_dim)
        for layer in self.layers:
            z = layer(z)
        actor_out = self.actor(z).view(-1, self.actor_out_dim)
        logits = actor_out[..., :self.action_dim]
        ab = actor_out[..., self.action_dim:]
        alpha, beta = _split_alpha_beta(ab)
        values = self.v_head(z).view(-1)
        return logits, alpha, beta, values

    def forward_seq(self, x_seq):
        z_seq = x_seq
        for layer in self.layers:
            z_seq = layer.forward_seq(z_seq)
        actor_out = self.actor.forward_seq(z_seq)  # (T, [B,] A+2)
        logits = actor_out[..., :self.action_dim]
        ab = actor_out[..., self.action_dim:]
        alpha, beta = _split_alpha_beta(ab)
        values = self.v_head.forward_seq(z_seq).squeeze(-1)  # (T, [B,])
        return logits, alpha, beta, values

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        trunk = [cell.get_state(clone=clone, detach=detach) for cell in self.layers]
        actor = self.actor.get_states(clone=clone, detach=detach)
        v_head = self.v_head.get_states(clone=clone, detach=detach)
        return {"trunk": trunk, "actor": actor, "v": v_head}

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
        self.v_head.set_states(v_states, clone=clone, detach=detach, strict=strict)


__all__ = [
    "BETA_EPS",
    "HybridActionValueNetwork",
    "RecurrentHybridActionValueNetwork",
]
