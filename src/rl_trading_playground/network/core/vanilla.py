"""Vanilla utilities for neural network architectures and reusable model components."""
from collections.abc import Callable
import torch


class VanillaNetwork(torch.nn.Module):

    """
    Fully connected multilayer perceptron.

    For hidden widths $d_1,\\dots,d_L$, the network maps an input vector $x$ by
    repeated affine transformations and nonlinearities:
    \\[
        h_0 = x,\\qquad h_\\ell = \\phi(W_\\ell h_{\\ell-1} + b_\\ell),
        \\quad \\ell=1,\\dots,L.
    \\]
    The output layer applies
    \\[
        y = g(W_o h_L + b_o),
    \\]
    where $g$ is ``out_act``.
    """
    def __init__(self, n_in, n_out, hidden_dims: list = [], activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh, out_act=lambda x: x) -> None:
        """Initialize the instance.

        Args:
            n_in (Any): The n in value.
            n_out (Any): The n out value.
            hidden_dims (list): The hidden dims value. Defaults to ``[]``.
            activation (Callable[[torch.Tensor], torch.Tensor]): The activation value. Defaults to ``torch.tanh``.
            out_act (Any): The out act value. Defaults to ``lambda x: x``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.out_act = out_act

        # create layers
        self.layers = torch.nn.ModuleList()
        prev = n_in
        for h in hidden_dims:
            self.layers.append(torch.nn.Linear(prev, h))
            prev = h
        self.out = torch.nn.Linear(prev, n_out)

    def forward(self, x):
        """Compute the forward pass.

        Args:
            x (Any): The x value.

        Returns:
            Any: The computed or requested result.
        """
        z = x.view(-1, self.n_in)
        for layer in self.layers:
            z = self.activation(layer(z))
        return self.out_act(self.out(z))

    def reset(self, batch_size: int = 1, device=None, dtype=None) -> None:
        """Reset internal state for a new episode or stream.

        Args:
            batch_size (int): The batch size value. Defaults to ``1``.
            device (Any): The device value. Defaults to ``None``.
            dtype (Any): The dtype value. Defaults to ``None``.

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
        shape = x_seq.shape
        y = self.forward(x_seq.reshape(-1, self.n_in))
        return y.reshape(*shape[:-1], self.n_out)

    def get_states(self, clone: bool = True, detach: bool = True):
        """Return a snapshot of recurrent states.

        Args:
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.

        Returns:
            Any: The computed or requested result.
        """
        return {}

    def set_states(self, states, clone: bool = True, detach: bool = True, strict: bool = True) -> None:
        """Restore recurrent states from a snapshot.

        Args:
            states (Any): The states value.
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.
            strict (bool): The strict value. Defaults to ``True``.

        Returns:
            None: This function does not return a value.
        """
        if strict and states not in ({}, None):
            raise ValueError("VanillaNetwork has no recurrent state.")


class ActionValueNetwork(torch.nn.Module):

    """
    Feedforward actor-critic network with a shared trunk.

    The shared encoder computes $z=f_\\theta(s)$ from state $s$. Two heads then
    produce policy logits and a scalar value estimate:
    \\[
        \\pi_{\\text{logits}}(s) = A_\\psi(z), \\qquad
        V(s) = V_\\omega(z).
    \\]
    This is the standard actor-critic factorization used by policy-gradient
    agents.
    """
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dims: list = [],
            hidden_dims_actor: list = [],
            hidden_dims_value: list = [],
            activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu
        ) -> None:
        """Initialize the instance.

        Args:
            state_dim (Any): The state dim value.
            action_dim (Any): The action dim value.
            hidden_dims (list): The hidden dims value. Defaults to ``[]``.
            hidden_dims_actor (list): The hidden dims actor value. Defaults to ``[]``.
            hidden_dims_value (list): The hidden dims value value. Defaults to ``[]``.
            activation (Callable[[torch.Tensor], torch.Tensor]): The activation value. Defaults to ``torch.relu``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.activation = activation

        # create layers
        self.layers = torch.nn.ModuleList()
        prev = state_dim
        for h in hidden_dims:
            self.layers.append(torch.nn.Linear(prev, h))
            prev = h

        # create actor head
        self.actor = VanillaNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_actor,
            activation=activation
        )

        # create value head
        self.value = VanillaNetwork(
            prev,
            1,
            hidden_dims=hidden_dims_value,
            activation=activation
        )

    def forward(self, x):
        """Compute the forward pass.

        Args:
            x (Any): The x value.

        Returns:
            Any: The computed or requested result.
        """
        z = x.view(-1, self.state_dim)
        for layer in self.layers:
            z = self.activation(layer(z))
        return z

    def a(self, x):
        """A for ActionValueNetwork.

        Args:
            x (Any): The x value.

        Returns:
            Any: The computed or requested result.
        """
        return self.actor(self.forward(x))

    def v(self, x):
        """V for ActionValueNetwork.

        Args:
            x (Any): The x value.

        Returns:
            Any: The computed or requested result.
        """
        return self.value(self.forward(x))

    def av(self, x):
        """Av for ActionValueNetwork.

        Args:
            x (Any): The x value.

        Returns:
            Any: The computed or requested result.
        """
        z = self.forward(x)
        return self.actor(z), self.value(z)

    def reset(self, batch_size: int = 1, device=None, dtype=None) -> None:
        """Reset internal state for a new episode or stream.

        Args:
            batch_size (int): The batch size value. Defaults to ``1``.
            device (Any): The device value. Defaults to ``None``.
            dtype (Any): The dtype value. Defaults to ``None``.

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
        shape = x_seq.shape
        z = self.forward(x_seq.reshape(-1, self.state_dim))
        actor = self.actor(z).reshape(*shape[:-1], self.action_dim)
        value = self.value(z).reshape(*shape[:-1], 1)
        return actor, value

    def get_states(self, clone: bool = True, detach: bool = True):
        """Return a snapshot of recurrent states.

        Args:
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.

        Returns:
            Any: The computed or requested result.
        """
        return {}

    def set_states(self, states, clone: bool = True, detach: bool = True, strict: bool = True) -> None:
        """Restore recurrent states from a snapshot.

        Args:
            states (Any): The states value.
            clone (bool): The clone value. Defaults to ``True``.
            detach (bool): The detach value. Defaults to ``True``.
            strict (bool): The strict value. Defaults to ``True``.

        Returns:
            None: This function does not return a value.
        """
        if strict and states not in ({}, None):
            raise ValueError("ActionValueNetwork has no recurrent state.")


class VanillaQNetwork(VanillaNetwork):

    """
    Feedforward action-value network for state-action pairs.

    The network concatenates state $s$ and action $a$, then approximates
    \\[
        Q(s,a) \\approx f_\\theta([s,a]).
    \\]
    It inherits the multilayer perceptron stack from ``VanillaNetwork`` and
    sets the input dimension to ``state_dim + action_dim``.
    """
    def __init__(self, state_dim, action_dim, hidden_dims: list = [], activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh, out_act=lambda x: x) -> None:
        """Initialize the instance.

        Args:
            state_dim (Any): The state dim value.
            action_dim (Any): The action dim value.
            hidden_dims (list): The hidden dims value. Defaults to ``[]``.
            activation (Callable[[torch.Tensor], torch.Tensor]): The activation value. Defaults to ``torch.tanh``.
            out_act (Any): The out act value. Defaults to ``lambda x: x``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(state_dim + action_dim, 1, hidden_dims, activation, out_act)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x, a):
        """Compute the forward pass.

        Args:
            x (Any): The x value.
            a (Any): The a value.

        Returns:
            Any: The computed or requested result.
        """
        xa = torch.cat([x, a], dim=-1)
        return super().forward(xa)
