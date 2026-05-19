import torch

from src.network.core.vanilla import VanillaNetwork


class FeedForwardActionValueNetwork(torch.nn.Module):
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
        shape = x.shape
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


class FeedForwardActionQNetwork(torch.nn.Module):
    """Shared feedforward trunk with categorical-policy and discrete-Q heads."""

    name = "action_q"

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=None,
        hidden_dims_actor=None,
        hidden_dims_q=None,
        activation=torch.relu,
    ):
        super().__init__()
        hidden_dims = hidden_dims or []
        hidden_dims_actor = hidden_dims_actor or []
        hidden_dims_q = hidden_dims_q or []

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
