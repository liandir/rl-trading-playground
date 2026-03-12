import math
import torch

RecurrentState = torch.Tensor | dict[str, torch.Tensor]


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


class LMUCell(torch.nn.Module):
    """
    Legendre Memory Unit cell with explicit internal state.

    State:
      - memory state m_t : shape (batch, memory_size)
      - hidden state h_t : shape (batch, hidden_size)

    Update:
      u_t = W_u x_t + W_hu h_{t-1}
      m_t = A_d m_{t-1} + B_d u_t
      h_t = act(W_x x_t + W_m m_t + W_h h_{t-1} + b)

    Notes
    -----
    - A_d, B_d are fixed buffers computed from the continuous-time LMU system
      and discretized with zero-order hold.
    - This is the standard "memory + nonlinear hidden state" LMU style.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        memory_size: int,
        theta: float,
        act=torch.tanh,
        learn_a_b: bool = False,
    ):
        super().__init__()
        if memory_size <= 0:
            raise ValueError("memory_size must be positive.")
        if theta <= 0:
            raise ValueError("theta must be positive.")

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.memory_size = int(memory_size)
        self.theta = float(theta)
        self.act = act

        # Input that drives the LMU memory
        self.w_u = torch.nn.Linear(n_in, 1, bias=False)
        self.w_hu = torch.nn.Linear(n_out, 1, bias=False)

        # Hidden update
        self.w_x = torch.nn.Linear(n_in, n_out, bias=False)
        self.w_m = torch.nn.Linear(memory_size, n_out, bias=False)
        self.w_h = torch.nn.Linear(n_out, n_out, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(n_out))

        # Explicit states
        self.memory_state = None
        self.hidden_state = None

        # Build continuous-time LMU matrices, then discretize
        A, B = self._make_continuous_lmu_matrices(memory_size, theta)

        if learn_a_b:
            self.A = torch.nn.Parameter(A)
            self.B = torch.nn.Parameter(B)
        else:
            self.register_buffer("A", A)
            self.register_buffer("B", B)

        A_d, B_d = self._discretize_zoh(A, B)
        if learn_a_b:
            self.A_d = torch.nn.Parameter(A_d)
            self.B_d = torch.nn.Parameter(B_d)
        else:
            self.register_buffer("A_d", A_d)
            self.register_buffer("B_d", B_d)

        torch.nn.init.orthogonal_(self.w_h.weight)
        torch.nn.init.xavier_uniform_(self.w_x.weight)
        torch.nn.init.xavier_uniform_(self.w_m.weight)
        torch.nn.init.xavier_uniform_(self.w_u.weight)
        torch.nn.init.xavier_uniform_(self.w_hu.weight)

    @staticmethod
    def _make_continuous_lmu_matrices(memory_size: int, theta: float):
        """
        Continuous-time LMU matrices from the Legendre delay system.

        A[i,j] = (2i+1)/theta * (-1 if i < j else (-1)^(i-j+1))
        B[i]   = (2i+1)/theta * (-1)^i

        Equivalent compact form used in many LMU implementations:
          A[i,j] = (2i+1)/theta * (-1 if i < j else (-1)^(i-j+1))
        which can be written via parity tests.
        """
        q = memory_size
        A = torch.zeros(q, q, dtype=torch.float32)
        B = torch.zeros(q, 1, dtype=torch.float32)

        for i in range(q):
            B[i, 0] = ((2 * i + 1) / theta) * ((-1.0) ** i)
            for j in range(q):
                if i < j:
                    A[i, j] = -((2 * i + 1) / theta)
                else:
                    A[i, j] = ((2 * i + 1) / theta) * ((-1.0) ** (i - j + 1))
        return A, B

    @staticmethod
    def _discretize_zoh(A: torch.Tensor, B: torch.Tensor, dt: float = 1.0):
        """
        Zero-order hold discretization using matrix exponential:

          [A_d  B_d] = exp([A B; 0 0] dt)
          [ 0    I ]

        Returns:
          A_d: (q, q)
          B_d: (q, 1)
        """
        q = A.shape[0]
        device = A.device
        dtype = A.dtype

        M = torch.zeros(q + 1, q + 1, device=device, dtype=dtype)
        M[:q, :q] = A
        M[:q, q:] = B
        Md = torch.matrix_exp(M * dt)

        A_d = Md[:q, :q]
        B_d = Md[:q, q:]
        return A_d, B_d

    def reset(
        self,
        batch_size: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        device = device if device is not None else self.b.device
        dtype = dtype if dtype is not None else self.b.dtype
        self.memory_state = torch.zeros(batch_size, self.memory_size, device=device, dtype=dtype)
        self.hidden_state = torch.zeros(batch_size, self.n_out, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hidden_state is None or self.memory_state is None or self.hidden_state.shape[0] != x.shape[0]:
            self.reset(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        # Scalar driving signal for memory
        u = self.w_u(x) + self.w_hu(self.hidden_state)  # (B, 1)

        # Linear memory update
        m_prev = self.memory_state
        m = m_prev @ self.A_d.T + u @ self.B_d.T

        # Nonlinear hidden update
        z = self.w_x(x) + self.w_m(m) + self.w_h(self.hidden_state) + self.b
        h = self.act(z)

        self.memory_state = m
        self.hidden_state = h
        return h

    def get_state(self, clone: bool = True, detach: bool = True) -> dict[str, torch.Tensor]:
        if self.memory_state is None or self.hidden_state is None:
            empty = torch.empty(0, device=self.b.device, dtype=self.b.dtype)
            return {"memory": empty, "hidden": empty}

        m = self.memory_state
        h = self.hidden_state

        if detach:
            m = m.detach()
            h = h.detach()
        if clone:
            m = m.clone()
            h = h.clone()

        return {"memory": m, "hidden": h}

    def set_state(self, state: dict[str, torch.Tensor], clone: bool = True, detach: bool = True):
        if not isinstance(state, dict):
            raise TypeError("state must be a dict with keys 'memory' and 'hidden'.")

        m = state.get("memory", None)
        h = state.get("hidden", None)

        if m is None or h is None or m.numel() == 0 or h.numel() == 0:
            self.memory_state = None
            self.hidden_state = None
            return

        if detach:
            m = m.detach()
            h = h.detach()
        if clone:
            m = m.clone()
            h = h.clone()

        self.memory_state = m.to(device=self.b.device, dtype=self.b.dtype)
        self.hidden_state = h.to(device=self.b.device, dtype=self.b.dtype)


def _build_recurrent_cell(
    n_in,
    n_out,
    activation,
    recurrent_type: str,
    recurrent_kwargs: dict | None = None,
):
    mode = (recurrent_type or "simple").lower()
    if mode in ("simple", "rnn", "recurrent"):
        return RecurrentCell(n_in, n_out, act=activation)
    if mode in ("gru", "gru_cell", "grucell"):
        return GRURecurrentCell(n_in, n_out)
    if mode in ("lmu", "lmu_cell", "lmucell"):
        kwargs = dict(recurrent_kwargs or {})
        memory_size = int(kwargs.pop("memory_size", n_out))
        theta = float(kwargs.pop("theta", memory_size))
        learn_a_b = bool(kwargs.pop("learn_a_b", False))
        act = kwargs.pop("act", activation)
        if kwargs:
            invalid = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported LMU recurrent_kwargs: {invalid}.")
        return LMUCell(
            n_in,
            n_out,
            memory_size=memory_size,
            theta=theta,
            act=act,
            learn_a_b=learn_a_b,
        )
    raise ValueError(f"Unsupported recurrent_type '{recurrent_type}'. Use 'simple', 'gru', or 'lmu'.")


class RecurrentNetwork(torch.nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        hidden_dims=None,
        activation=torch.tanh,
        out_act=torch.nn.Identity(),
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or []
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.out_act = out_act
        self.recurrent_type = recurrent_type
        self.recurrent_kwargs = dict(recurrent_kwargs or {})

        self.cells = torch.nn.ModuleList()
        prev = n_in
        for h in hidden_dims:
            self.cells.append(_build_recurrent_cell(prev, h, activation, recurrent_type, self.recurrent_kwargs))
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

    def get_states(self, clone: bool = True, detach: bool = True) -> list[RecurrentState]:
        return [cell.get_state(clone=clone, detach=detach) for cell in self.cells]

    def set_states(
        self,
        states: list[RecurrentState],
        clone: bool = True,
        detach: bool = True,
        strict: bool = True,
    ):
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
        recurrent_kwargs: dict | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or []
        hidden_dims_actor = hidden_dims_actor or []
        hidden_dims_value = hidden_dims_value or []

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
        self.value = RecurrentNetwork(
            prev,
            1,
            hidden_dims=hidden_dims_value,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=self.recurrent_kwargs,
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
