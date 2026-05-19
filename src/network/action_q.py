import torch

from src.network.core.recurrent import RecurrentNetwork, _build_recurrent_cell
from src.network.core.vanilla import VanillaNetwork


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

