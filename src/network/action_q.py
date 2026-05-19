"""Action q utilities for neural network architectures and reusable model components."""
from collections.abc import Callable
import torch

from src.network.core.recurrent import RecurrentNetwork, _build_recurrent_cell
from src.network.core.vanilla import VanillaNetwork


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
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            state_dim (Any): The state dim value.
            action_dim (Any): The action dim value.
            hidden_dims (Any): The hidden dims value. Defaults to ``None``.
            hidden_dims_actor (Any): The hidden dims actor value. Defaults to ``None``.
            hidden_dims_q (Any): The hidden dims q value. Defaults to ``None``.
            activation (Callable[[torch.Tensor], torch.Tensor]): The activation value. Defaults to ``torch.tanh``.
            recurrent_type (str): The recurrent type value. Defaults to ``'simple'``.
            recurrent_kwargs (dict | None): The recurrent kwargs value. Defaults to ``None``.

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

    def reset(self, batch_size: int = 1) -> None:
        """Reset internal state for a new episode or stream.

        Args:
            batch_size (int): The batch size value. Defaults to ``1``.

        Returns:
            None: This function does not return a value.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        for layer in self.layers:
            layer.reset(batch_size, device=device, dtype=dtype)
        self.actor.reset(batch_size, device=device, dtype=dtype)
        self.q_head.reset(batch_size, device=device, dtype=dtype)

    def forward(self, x):
        """Compute the forward pass.

        Args:
            x (Any): The x value.

        Returns:
            Any: The computed or requested result.
        """
        z = x.view(-1, self.state_dim)
        for layer in self.layers:
            z = layer(z)
        return self.actor(z), self.q_head(z)

    def forward_seq(self, x_seq):
        """Compute a forward pass over a sequence.

        Args:
            x_seq (Any): The x seq value.

        Returns:
            Any: The computed or requested result.
        """
        z_seq = x_seq
        for layer in self.layers:
            z_seq = layer.forward_seq(z_seq)
        return self.actor.forward_seq(z_seq), self.q_head.forward_seq(z_seq)

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        """Return a snapshot of recurrent states.

        Args:
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.

        Returns:
            dict: The computed or requested result.
        """
        trunk = [cell.get_state(clone=clone, detach=detach) for cell in self.layers]
        actor = self.actor.get_states(clone=clone, detach=detach)
        q_head = self.q_head.get_states(clone=clone, detach=detach)
        return {"trunk": trunk, "actor": actor, "q": q_head}

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
