import torch


class RecurrentCell(torch.nn.Module):
    def __init__(self, n_in, n_out, act=torch.tanh):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.act = act

        self.w_ext = torch.nn.Linear(n_in, n_out, bias=False)
        self.w_rec = torch.nn.Linear(n_out, n_out, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(n_out))
        self.state = None

        torch.nn.init.orthogonal_(self.w_rec.weight)
        torch.nn.init.orthogonal_(self.w_ext.weight)

    def reset(self, batch_size: int = 1, device: torch.device = None, dtype: torch.dtype = None):
        device = device if device is not None else self.b.device
        dtype = dtype if dtype is not None else self.b.dtype
        self.state = torch.zeros(batch_size, self.n_out, device=device, dtype=dtype)

    def forward(self, x):
        if self.state is None or self.state.shape[0] != x.shape[0]:
            self.reset(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        z = self.w_ext(x) + self.w_rec(self.state) + self.b
        self.state = self.act(z)
        return self.state

    def get_state(self, clone: bool = True, detach: bool = True) -> torch.Tensor:
        if self.state is None:
            return torch.empty(0, device=self.b.device, dtype=self.b.dtype)
        s = self.state
        if detach:
            s = s.detach()
        if clone:
            s = s.clone()
        return s

    def set_state(self, state: torch.Tensor, clone: bool = True, detach: bool = True):
        if state.numel() == 0:
            self.state = None
            return
        s = state
        if detach:
            s = s.detach()
        if clone:
            s = s.clone()
        self.state = s.to(device=self.b.device, dtype=self.b.dtype)


class GRURecurrentCell(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.cell = torch.nn.GRUCell(n_in, n_out)
        self.state = None

        torch.nn.init.xavier_uniform_(self.cell.weight_ih)
        torch.nn.init.orthogonal_(self.cell.weight_hh)
        torch.nn.init.zeros_(self.cell.bias_ih)
        torch.nn.init.zeros_(self.cell.bias_hh)

    def reset(self, batch_size: int = 1, device: torch.device = None, dtype: torch.dtype = None):
        device = device if device is not None else self.cell.weight_hh.device
        dtype = dtype if dtype is not None else self.cell.weight_hh.dtype
        self.state = torch.zeros(batch_size, self.n_out, device=device, dtype=dtype)

    def forward(self, x):
        if self.state is None or self.state.shape[0] != x.shape[0]:
            self.reset(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        self.state = self.cell(x, self.state)
        return self.state

    def get_state(self, clone: bool = True, detach: bool = True) -> torch.Tensor:
        if self.state is None:
            return torch.empty(0, device=self.cell.weight_hh.device, dtype=self.cell.weight_hh.dtype)
        s = self.state
        if detach:
            s = s.detach()
        if clone:
            s = s.clone()
        return s

    def set_state(self, state: torch.Tensor, clone: bool = True, detach: bool = True):
        if state.numel() == 0:
            self.state = None
            return
        s = state
        if detach:
            s = s.detach()
        if clone:
            s = s.clone()
        self.state = s.to(device=self.cell.weight_hh.device, dtype=self.cell.weight_hh.dtype)


def _build_recurrent_cell(n_in, n_out, activation, recurrent_type: str):
    mode = (recurrent_type or "simple").lower()
    if mode in ("simple", "rnn", "recurrent"):
        return RecurrentCell(n_in, n_out, act=activation)
    if mode in ("gru", "gru_cell", "grucell"):
        return GRURecurrentCell(n_in, n_out)
    raise ValueError(f"Unsupported recurrent_type '{recurrent_type}'. Use 'simple' or 'gru'.")


class RecurrentNetwork(torch.nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        hidden_dims=None,
        activation=torch.tanh,
        out_act=torch.nn.Identity(),
        recurrent_type: str = "simple",
    ):
        super().__init__()
        hidden_dims = hidden_dims or []
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.out_act = out_act
        self.recurrent_type = recurrent_type

        self.cells = torch.nn.ModuleList()
        prev = n_in
        for h in hidden_dims:
            self.cells.append(_build_recurrent_cell(prev, h, activation, recurrent_type))
            prev = h
        self.out = torch.nn.Linear(prev, n_out)

    def reset(self, batch_size, device=None, dtype=None):
        dev = device if device is not None else self.out.weight.device
        dt = dtype if dtype is not None else self.out.weight.dtype
        for cell in self.cells:
            cell.reset(batch_size, device=dev, dtype=dt)

    def forward(self, x):
        z = x.view(-1, self.n_in)
        for cell in self.cells:
            z = cell(z)
        return self.out_act(self.out(z))

    def get_states(self, clone: bool = True, detach: bool = True) -> list[torch.Tensor]:
        return [cell.get_state(clone=clone, detach=detach) for cell in self.cells]

    def set_states(self, states: list[torch.Tensor], clone: bool = True, detach: bool = True, strict: bool = True):
        states = states or []
        if strict and len(states) != len(self.cells):
            raise ValueError(f"Expected {len(self.cells)} cell states, got {len(states)}.")
        for cell, s in zip(self.cells, states):
            cell.set_state(s, clone=clone, detach=detach)


class RecurrentActionValueNetwork(torch.nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=None,
        hidden_dims_actor=None,
        hidden_dims_value=None,
        activation=torch.tanh,
        recurrent_type: str = "simple",
    ):
        super().__init__()
        hidden_dims = hidden_dims or []
        hidden_dims_actor = hidden_dims_actor or []
        hidden_dims_value = hidden_dims_value or []

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = activation
        self.recurrent_type = recurrent_type

        self.layers = torch.nn.ModuleList()
        prev = state_dim
        for h in hidden_dims:
            self.layers.append(_build_recurrent_cell(prev, h, activation, recurrent_type))
            prev = h

        self.actor = RecurrentNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_actor,
            activation=activation,
            recurrent_type=recurrent_type,
        )
        self.value = RecurrentNetwork(
            prev,
            1,
            hidden_dims=hidden_dims_value,
            activation=activation,
            recurrent_type=recurrent_type,
        )

    def reset(self, batch_size: int = 1):
        dev = next(self.parameters()).device
        dt = next(self.parameters()).dtype
        for cell in self.layers:
            cell.reset(batch_size, device=dev, dtype=dt)
        self.actor.reset(batch_size, device=dev, dtype=dt)
        self.value.reset(batch_size, device=dev, dtype=dt)

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

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        trunk = [cell.get_state(clone=clone, detach=detach) for cell in self.layers]
        actor = self.actor.get_states(clone=clone, detach=detach)
        value = self.value.get_states(clone=clone, detach=detach)
        return {"trunk": trunk, "actor": actor, "value": value}

    def set_states(self, states: dict, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict:
            for k in ("trunk", "actor", "value"):
                if k not in states:
                    raise KeyError(f"Missing key '{k}' in states snapshot.")

        trunk_states = states.get("trunk", [])
        actor_states = states.get("actor", [])
        value_states = states.get("value", [])

        if strict and len(trunk_states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} trunk states, got {len(trunk_states)}.")

        for cell, s in zip(self.layers, trunk_states):
            cell.set_state(s, clone=clone, detach=detach)

        self.actor.set_states(actor_states, clone=clone, detach=detach, strict=strict)
        self.value.set_states(value_states, clone=clone, detach=detach, strict=strict)
