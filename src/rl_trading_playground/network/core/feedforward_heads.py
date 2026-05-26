"""Feedforward heads utilities for neural network architectures and reusable model components."""
from collections.abc import Callable
import torch

from rl_trading_playground.network.core.vanilla import VanillaNetwork


class FeedForwardActionValueNetwork(torch.nn.Module):
    """
    Shared feedforward trunk with categorical-policy and scalar-value heads.

    The trunk maps state $s$ to latent feature $z=f_\\theta(s)$. The actor and
    critic heads then compute
    \\[
        \\ell(s) = A_\\psi(z), \\qquad V(s) = C_\\omega(z),
    \\]
    where $\\ell(s)$ are unnormalized action logits and $V(s)$ is a scalar
    value estimate.
    """

    name = "action_value"

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=None,
        hidden_dims_actor=None,
        hidden_dims_value=None,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
    ) -> None:
        """Initialize the instance.

        Args:
            state_dim (Any): The state dim value.
            action_dim (Any): The action dim value.
            hidden_dims (Any): The hidden dims value. Defaults to ``None``.
            hidden_dims_actor (Any): The hidden dims actor value. Defaults to ``None``.
            hidden_dims_value (Any): The hidden dims value value. Defaults to ``None``.
            activation (Callable[[torch.Tensor], torch.Tensor]): The activation value. Defaults to ``torch.relu``.

        Returns:
            None: This function does not return a value.
        """
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
        """Compute the forward pass.

        Args:
            x (Any): The x value.

        Returns:
            Any: The computed or requested result.
        """
        shape = x.shape
        z = x.reshape(-1, self.state_dim)
        for layer in self.layers:
            z = self.activation(layer(z))
        logits = self.actor(z).view(*shape[:-1], self.action_dim)
        values = self.value(z).squeeze(-1).view(*shape[:-1])
        return logits, values

    def reset(self, batch_size: int = 1) -> None:
        """Reset internal state for a new episode or stream.

        Args:
            batch_size (int): The batch size value. Defaults to ``1``.

        Returns:
            Any: The computed or requested result.
        """
        return None

    def forward_seq(self, x_seq):
        """Compute a forward pass over a sequence.

        Args:
            x_seq (Any): The x seq value.

        Returns:
            Any: The computed or requested result.
        """
        return self.forward(x_seq)

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        """Return a snapshot of recurrent states.

        Args:
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.

        Returns:
            dict: The computed or requested result.
        """
        return {}

    def set_states(self, states: dict | None, clone: bool = True, detach: bool = True, strict: bool = True) -> None:
        """Restore recurrent states from a snapshot.

        Args:
            states (dict | None): The states value.
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.
            strict (bool): The strict value. Defaults to ``True``.

        Returns:
            None: This function does not return a value.
        """
        if strict and states not in ({}, None):
            raise ValueError("Feedforward ActionValueNetwork has no recurrent state.")


class FeedForwardActionQNetwork(torch.nn.Module):
    """
    Shared feedforward trunk with categorical-policy and discrete-Q heads.

    From state $s$, the shared trunk forms $z=f_\\theta(s)$. The policy head
    produces logits $\\ell(s)$, while the Q head estimates one value per
    discrete action:
    \\[
        Q(s, a_i) = Q_\\omega(z)_i,\\qquad i=1,\\dots,|\\mathcal{A}|.
    \\]
    """

    name = "action_q"

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=None,
        hidden_dims_actor=None,
        hidden_dims_q=None,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
    ) -> None:
        """Initialize the instance.

        Args:
            state_dim (Any): The state dim value.
            action_dim (Any): The action dim value.
            hidden_dims (Any): The hidden dims value. Defaults to ``None``.
            hidden_dims_actor (Any): The hidden dims actor value. Defaults to ``None``.
            hidden_dims_q (Any): The hidden dims q value. Defaults to ``None``.
            activation (Callable[[torch.Tensor], torch.Tensor]): The activation value. Defaults to ``torch.relu``.

        Returns:
            None: This function does not return a value.
        """
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
        """Compute the forward pass.

        Args:
            x (Any): The x value.

        Returns:
            Any: The computed or requested result.
        """
        shape = x.shape
        z = x.view(-1, self.state_dim)
        for layer in self.layers:
            z = self.activation(layer(z))
        logits = self.actor(z).view(*shape[:-1], self.action_dim)
        q_values = self.q_head(z).view(*shape[:-1], self.action_dim)
        return logits, q_values

    def reset(self, batch_size: int = 1) -> None:
        """Reset internal state for a new episode or stream.

        Args:
            batch_size (int): The batch size value. Defaults to ``1``.

        Returns:
            Any: The computed or requested result.
        """
        return None

    def forward_seq(self, x_seq):
        """Compute a forward pass over a sequence.

        Args:
            x_seq (Any): The x seq value.

        Returns:
            Any: The computed or requested result.
        """
        return self.forward(x_seq)

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        """Return a snapshot of recurrent states.

        Args:
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.

        Returns:
            dict: The computed or requested result.
        """
        return {}

    def set_states(self, states: dict | None, clone: bool = True, detach: bool = True, strict: bool = True) -> None:
        """Restore recurrent states from a snapshot.

        Args:
            states (dict | None): The states value.
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.
            strict (bool): The strict value. Defaults to ``True``.

        Returns:
            None: This function does not return a value.
        """
        if strict and states not in ({}, None):
            raise ValueError("Feedforward ActionQNetwork has no recurrent state.")
