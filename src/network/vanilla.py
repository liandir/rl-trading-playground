import torch


class VanillaNetwork(torch.nn.Module):

    def __init__(self, n_in, n_out, hidden_dims=[], activation=torch.tanh, out_act=lambda x: x):
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
        z = x.view(-1, self.n_in)
        for layer in self.layers:
            z = self.activation(layer(z))
        return self.out_act(self.out(z))

    def reset(self, batch_size: int = 1, device=None, dtype=None):
        return None

    def forward_seq(self, x_seq):
        shape = x_seq.shape
        y = self.forward(x_seq.reshape(-1, self.n_in))
        return y.reshape(*shape[:-1], self.n_out)

    def get_states(self, clone: bool = True, detach: bool = True):
        return {}

    def set_states(self, states, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict and states not in ({}, None):
            raise ValueError("VanillaNetwork has no recurrent state.")
    

class ActionValueNetwork(torch.nn.Module):

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dims=[],
            hidden_dims_actor = [],
            hidden_dims_value = [],
            activation=torch.relu
        ):
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
        z = x.view(-1, self.state_dim)
        for layer in self.layers:
            z = self.activation(layer(z))
        return z
    
    def a(self, x):
        return self.actor(self.forward(x))
    
    def v(self, x):
        return self.value(self.forward(x))
    
    def av(self, x):
        z = self.forward(x)
        return self.actor(z), self.value(z)

    def reset(self, batch_size: int = 1, device=None, dtype=None):
        return None

    def forward_seq(self, x_seq):
        shape = x_seq.shape
        z = self.forward(x_seq.reshape(-1, self.state_dim))
        actor = self.actor(z).reshape(*shape[:-1], self.action_dim)
        value = self.value(z).reshape(*shape[:-1], 1)
        return actor, value

    def get_states(self, clone: bool = True, detach: bool = True):
        return {}

    def set_states(self, states, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict and states not in ({}, None):
            raise ValueError("ActionValueNetwork has no recurrent state.")


class VanillaQNetwork(VanillaNetwork):

    def __init__(self, state_dim, action_dim, hidden_dims=[], activation=torch.tanh, out_act=lambda x: x):
        super().__init__(state_dim + action_dim, 1, hidden_dims, activation, out_act)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x, a):
        xa = torch.cat([x, a], dim=-1)
        return super().forward(xa)

