import math
import torch

RecurrentState = torch.Tensor | dict[str, torch.Tensor]


def _normalize_sequence_input(
    x_seq: torch.Tensor,
    *,
    expected_in: int,
) -> tuple[torch.Tensor, bool]:
    """
    Normalize sequence input to (T, B, input_dim).

    Accepts either an unbatched sequence (T, input_dim) or an explicitly
    batched sequence (T, B, input_dim). Returns the normalized tensor plus a
    flag indicating whether the caller should squeeze the singleton batch axis
    back out of the output.
    """
    if x_seq.dim() == 2:
        if x_seq.shape[-1] != expected_in:
            raise ValueError(f"Expected input dim {expected_in}, got {x_seq.shape[-1]}.")
        return x_seq.unsqueeze(1), True

    if x_seq.dim() == 3:
        if x_seq.shape[-1] != expected_in:
            raise ValueError(f"Expected input dim {expected_in}, got {x_seq.shape[-1]}.")
        return x_seq, False

    raise ValueError(
        f"Expected sequence input with shape (T, {expected_in}) or (T, B, {expected_in}), "
        f"got {tuple(x_seq.shape)}."
    )


def _copy_trace_tensor(
    value: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    clone: bool = True,
    detach: bool = True,
) -> torch.Tensor:
    if value is None:
        return torch.empty(0, device=device, dtype=dtype)
    out = value
    if detach:
        out = out.detach()
    if clone:
        out = out.clone()
    return out.to(device=device, dtype=dtype)


def _restore_trace_tensor(
    value: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    clone: bool = True,
    detach: bool = True,
) -> torch.Tensor | None:
    if value is None or value.numel() == 0:
        return None
    out = value
    if detach:
        out = out.detach()
    if clone:
        out = out.clone()
    return out.to(device=device, dtype=dtype)


def _collect_plastic_modules(modules) -> list[torch.nn.Module]:
    return [module for module in modules if hasattr(module, "plastic_parameters") and hasattr(module, "apply_online_gradients")]


def _apply_online_updates(
    modules,
    loss: torch.Tensor,
    *,
    retain_graph: bool = False,
) -> bool:
    plastic_modules = _collect_plastic_modules(modules)
    if not plastic_modules:
        return False

    loss_scalar = loss.mean() if loss.dim() > 0 else loss
    params = []
    counts = []
    for module in plastic_modules:
        module_params = tuple(module.plastic_parameters())
        if not module_params:
            continue
        params.extend(module_params)
        counts.append((module, len(module_params)))

    if not params:
        return False

    grads = torch.autograd.grad(loss_scalar, params, retain_graph=retain_graph, allow_unused=True)
    error_t = loss_scalar.detach()

    offset = 0
    for module, count in counts:
        module.apply_online_gradients(grads[offset:offset + count], error_t=error_t)
        offset += count

    return True


class RecurrentCell(torch.nn.Module):
    _ACT_TO_NONLINEARITY = {torch.tanh: "tanh", torch.relu: "relu"}

    def __init__(self, n_in, n_out, act=torch.tanh):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.act = act

        nonlinearity = self._ACT_TO_NONLINEARITY.get(act)
        if nonlinearity is None:
            raise ValueError(
                f"RecurrentCell only supports torch.tanh or torch.relu; got {act}."
            )
        self.rnn = torch.nn.RNN(input_size=n_in, hidden_size=n_out, batch_first=False, nonlinearity=nonlinearity)
        self.state = None

        torch.nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        torch.nn.init.orthogonal_(self.rnn.weight_hh_l0)
        torch.nn.init.zeros_(self.rnn.bias_ih_l0)
        torch.nn.init.zeros_(self.rnn.bias_hh_l0)

    def reset(self, batch_size: int = 1, device: torch.device = None, dtype: torch.dtype = None):
        device = device if device is not None else self.rnn.weight_hh_l0.device
        dtype = dtype if dtype is not None else self.rnn.weight_hh_l0.dtype
        self.state = torch.zeros(batch_size, self.n_out, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step forward.  x: (B, n_in)  →  (B, n_out)"""
        B = x.shape[0]
        if self.state is None or self.state.shape[0] != B:
            self.reset(B, device=x.device, dtype=x.dtype)
        _, h_n = self.rnn(x.unsqueeze(0), self.state.unsqueeze(0))
        self.state = h_n.squeeze(0)
        return self.state

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Full-sequence forward using the fused RNN kernel.

        x_seq: (T, n_in)  →  (T, n_out)
        """
        x_seq_3d, squeeze_batch = _normalize_sequence_input(x_seq, expected_in=self.n_in)
        B = x_seq_3d.shape[1]
        if self.state is None or self.state.shape[0] != B:
            self.reset(B, device=x_seq.device, dtype=x_seq.dtype)
        out, h_n = self.rnn(x_seq_3d, self.state.unsqueeze(0))
        self.state = h_n.squeeze(0)
        return out.squeeze(1) if squeeze_batch else out

    def get_state(self, clone: bool = True, detach: bool = True) -> torch.Tensor:
        if self.state is None:
            return torch.empty(0, device=self.rnn.weight_hh_l0.device, dtype=self.rnn.weight_hh_l0.dtype)
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
        self.state = s.to(device=self.rnn.weight_hh_l0.device, dtype=self.rnn.weight_hh_l0.dtype)


class GradientOjaRecurrentCell(torch.nn.Module):
    """
    Recurrent cell with local Oja plasticity plus an EMA of task gradients.

    The integrated error state follows:
      E <- E + nu * (E_t - E)

    and the same EMA is applied to per-parameter gradients so the weight update
    can use a low-pass filtered task signal without backpropagating through the
    full online history.
    """

    def __init__(
        self,
        n_in,
        n_out,
        act=torch.tanh,
        eta: float = 1e-3,
        lambda_: float = 1e-4,
        alpha: float = 1.0,
        nu: float = 0.1,
        wclip: float | None = None,
    ):
        super().__init__()
        if nu <= 0:
            raise ValueError("nu must be positive.")
        if wclip is not None and wclip <= 0:
            raise ValueError("wclip must be positive when provided.")

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.act = act
        self.eta = float(eta)
        self.lambda_ = float(lambda_)
        self.alpha = float(alpha)
        self.nu = float(nu)
        self.wclip = None if wclip is None else float(wclip)

        self.w_ext = torch.nn.Linear(n_in, n_out, bias=False)
        self.w_rec = torch.nn.Linear(n_out, n_out, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(n_out))
        self.state = None

        self.register_buffer("error_ema", torch.zeros(()))
        self.register_buffer("grad_ema_ext", torch.zeros_like(self.w_ext.weight))
        self.register_buffer("grad_ema_rec", torch.zeros_like(self.w_rec.weight))
        self.register_buffer("grad_ema_bias", torch.zeros_like(self.b))

        self._last_input = None
        self._last_prev_state = None
        self._last_output = None

        torch.nn.init.orthogonal_(self.w_rec.weight)
        torch.nn.init.orthogonal_(self.w_ext.weight)

    def reset(self, batch_size: int = 1, device: torch.device = None, dtype: torch.dtype = None):
        device = device if device is not None else self.b.device
        dtype = dtype if dtype is not None else self.b.dtype
        self.state = torch.zeros(batch_size, self.n_out, device=device, dtype=dtype)
        self._last_input = None
        self._last_prev_state = None
        self._last_output = None

    def forward(self, x):
        if self.state is None or self.state.shape[0] != x.shape[0]:
            self.reset(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        prev_state = self.state
        z = self.w_ext(x) + self.w_rec(prev_state) + self.b
        y = self.act(z)

        self._last_input = x.detach()
        self._last_prev_state = prev_state.detach()
        self._last_output = y.detach()
        self.state = y
        return self.state

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (T, n_in)  →  (T, n_out)
        Loops over time internally; state and online traces are updated in-place.
        """
        x_seq_3d, squeeze_batch = _normalize_sequence_input(x_seq, expected_in=self.n_in)
        steps = [self(x_seq_3d[t]) for t in range(x_seq_3d.shape[0])]
        out = torch.stack(steps, dim=0)
        return out.squeeze(1) if squeeze_batch else out

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

    def get_online_trace(self, clone: bool = True, detach: bool = True) -> dict[str, torch.Tensor]:
        return {
            "x": _copy_trace_tensor(self._last_input, device=self.b.device, dtype=self.b.dtype, clone=clone, detach=detach),
            "rec": _copy_trace_tensor(self._last_prev_state, device=self.b.device, dtype=self.b.dtype, clone=clone, detach=detach),
            "y": _copy_trace_tensor(self._last_output, device=self.b.device, dtype=self.b.dtype, clone=clone, detach=detach),
        }

    def set_online_trace(self, trace: dict[str, torch.Tensor] | None, clone: bool = True, detach: bool = True):
        trace = trace or {}
        self._last_input = _restore_trace_tensor(trace.get("x"), device=self.b.device, dtype=self.b.dtype, clone=clone, detach=detach)
        self._last_prev_state = _restore_trace_tensor(trace.get("rec"), device=self.b.device, dtype=self.b.dtype, clone=clone, detach=detach)
        self._last_output = _restore_trace_tensor(trace.get("y"), device=self.b.device, dtype=self.b.dtype, clone=clone, detach=detach)

    def plastic_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return (self.w_ext.weight, self.w_rec.weight, self.b)

    def _update_grad_ema(self, ema: torch.Tensor, grad: torch.Tensor | None):
        if grad is None:
            ema.mul_(1.0 - self.nu)
            return
        grad_t = grad.detach().to(device=ema.device, dtype=ema.dtype)
        ema.add_(self.nu * (grad_t - ema))

    @staticmethod
    def _oja_term(
        y: torch.Tensor | None,
        x: torch.Tensor | None,
        weight: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        if y is None or x is None or y.numel() == 0 or x.numel() == 0:
            return torch.zeros_like(weight)
        proj = x @ weight.T
        term = y.unsqueeze(-1) * (alpha * x.unsqueeze(1) - proj.unsqueeze(-1) * weight.unsqueeze(0))
        return term.mean(dim=0)

    def apply_online_gradients(self, gradients, *, error_t: torch.Tensor):
        grad_ext, grad_rec, grad_bias = gradients

        with torch.no_grad():
            err = error_t.to(device=self.error_ema.device, dtype=self.error_ema.dtype)
            self.error_ema.add_(self.nu * (err - self.error_ema))

            self._update_grad_ema(self.grad_ema_ext, grad_ext)
            self._update_grad_ema(self.grad_ema_rec, grad_rec)
            self._update_grad_ema(self.grad_ema_bias, grad_bias)

            oja_ext = self._oja_term(self._last_output, self._last_input, self.w_ext.weight, self.alpha)
            oja_rec = self._oja_term(self._last_output, self._last_prev_state, self.w_rec.weight, self.alpha)

            self.w_ext.weight.add_(-self.eta * self.grad_ema_ext + self.lambda_ * oja_ext)
            self.w_rec.weight.add_(-self.eta * self.grad_ema_rec + self.lambda_ * oja_rec)
            self.b.add_(-self.eta * self.grad_ema_bias)

            if self.wclip is not None:
                self.w_ext.weight.clamp_(-self.wclip, self.wclip)
                self.w_rec.weight.clamp_(-self.wclip, self.wclip)


# ---------------------------------------------------------------------------
# GRURecurrentCell — uses nn.GRU (sequence mode) for CuDNN acceleration.
#
# Key difference from the old implementation:
#   - nn.GRUCell  processes one timestep at a time (Python loop required)
#   - nn.GRU      processes a full (T, B, input_size) tensor in a single
#                 fused CUDA kernel when running on GPU, giving a 10-50x
#                 speedup for sequences.
#
# The public API (reset / forward / get_state / set_state) is unchanged.
# forward_seq() is the new high-speed path used by RecurrentNetwork.forward_seq.
# ---------------------------------------------------------------------------
class GRURecurrentCell(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        # nn.GRU instead of nn.GRUCell — enables CuDNN sequence kernel
        self.gru = torch.nn.GRU(input_size=n_in, hidden_size=n_out, batch_first=False)
        self.state = None  # stored as (B, n_out); reshaped to (1, B, n_out) for GRU calls

        torch.nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        torch.nn.init.orthogonal_(self.gru.weight_hh_l0)
        torch.nn.init.zeros_(self.gru.bias_ih_l0)
        torch.nn.init.zeros_(self.gru.bias_hh_l0)

    def reset(self, batch_size: int = 1, device: torch.device = None, dtype: torch.dtype = None):
        device = device if device is not None else self.gru.weight_hh_l0.device
        dtype = dtype if dtype is not None else self.gru.weight_hh_l0.dtype
        self.state = torch.zeros(batch_size, self.n_out, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step forward.  x: (B, n_in)  →  (B, n_out)"""
        B = x.shape[0]
        if self.state is None or self.state.shape[0] != B:
            self.reset(B, device=x.device, dtype=x.dtype)
        # GRU expects (T=1, B, input_size) and h0 of (1, B, hidden_size)
        out, h_n = self.gru(x.unsqueeze(0), self.state.unsqueeze(0))
        self.state = h_n.squeeze(0)   # (B, n_out)
        return self.state

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Full-sequence forward using the CuDNN fused kernel.

        x_seq: (T, n_in)  →  (T, n_out)   [implicit B=1]

        The entire temporal loop runs inside a single CUDA kernel — no Python
        overhead per timestep.  self.state is updated to h_T so subsequent
        single-step forward() calls continue from the correct hidden state.
        """
        # Add and later remove the batch dimension expected by nn.GRU
        x_seq_3d, squeeze_batch = _normalize_sequence_input(x_seq, expected_in=self.n_in)
        B = x_seq_3d.shape[1]
        if self.state is None or self.state.shape[0] != B:
            self.reset(B, device=x_seq.device, dtype=x_seq.dtype)
        h0 = self.state.unsqueeze(0)                           # (1, B, n_out)
        out, h_n = self.gru(x_seq_3d, h0)                     # out: (T, B, n_out)
        self.state = h_n.squeeze(0)                            # (1, n_out) — B=1
        return out.squeeze(1) if squeeze_batch else out

    def get_state(self, clone: bool = True, detach: bool = True) -> torch.Tensor:
        if self.state is None:
            return torch.empty(0, device=self.gru.weight_hh_l0.device, dtype=self.gru.weight_hh_l0.dtype)
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
        self.state = s.to(device=self.gru.weight_hh_l0.device, dtype=self.gru.weight_hh_l0.dtype)


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

        self.w_u = torch.nn.Linear(n_in, 1, bias=False)
        self.w_hu = torch.nn.Linear(n_out, 1, bias=False)
        self.w_x = torch.nn.Linear(n_in, n_out, bias=False)
        self.w_m = torch.nn.Linear(memory_size, n_out, bias=False)
        self.w_h = torch.nn.Linear(n_out, n_out, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(n_out))

        self.memory_state = None
        self.hidden_state = None

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
        q = A.shape[0]
        device = A.device
        dtype = A.dtype
        M = torch.zeros(q + 1, q + 1, device=device, dtype=dtype)
        M[:q, :q] = A
        M[:q, q:] = B
        Md = torch.matrix_exp(M * dt)
        return Md[:q, :q], Md[:q, q:]

    def reset(self, batch_size: int = 1, device: torch.device = None, dtype: torch.dtype = None):
        device = device if device is not None else self.b.device
        dtype = dtype if dtype is not None else self.b.dtype
        self.memory_state = torch.zeros(batch_size, self.memory_size, device=device, dtype=dtype)
        self.hidden_state = torch.zeros(batch_size, self.n_out, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hidden_state is None or self.memory_state is None or self.hidden_state.shape[0] != x.shape[0]:
            self.reset(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        u = self.w_u(x) + self.w_hu(self.hidden_state)
        m = self.memory_state @ self.A_d.T + u @ self.B_d.T
        z = self.w_x(x) + self.w_m(m) + self.w_h(self.hidden_state) + self.b
        h = self.act(z)
        self.memory_state = m
        self.hidden_state = h
        return h

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (T, n_in)  →  (T, n_out)
        Loops over time internally; both memory and hidden states are updated.
        """
        x_seq_3d, squeeze_batch = _normalize_sequence_input(x_seq, expected_in=self.n_in)
        steps = [self(x_seq_3d[t]) for t in range(x_seq_3d.shape[0])]
        out = torch.stack(steps, dim=0)
        return out.squeeze(1) if squeeze_batch else out

    def get_state(self, clone: bool = True, detach: bool = True) -> dict[str, torch.Tensor]:
        if self.memory_state is None or self.hidden_state is None:
            empty = torch.empty(0, device=self.b.device, dtype=self.b.dtype)
            return {"memory": empty, "hidden": empty}
        m, h = self.memory_state, self.hidden_state
        if detach:
            m, h = m.detach(), h.detach()
        if clone:
            m, h = m.clone(), h.clone()
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
            m, h = m.detach(), h.detach()
        if clone:
            m, h = m.clone(), h.clone()
        self.memory_state = m.to(device=self.b.device, dtype=self.b.dtype)
        self.hidden_state = h.to(device=self.b.device, dtype=self.b.dtype)


class ParallelRecurrentCell(torch.nn.Module):
    """
    RNN cell with a parallel linear (feedthrough) branch, merged by addition.

    At each timestep the two branches are computed independently and summed:

        h_t  = rec_act(W_ih x_t + W_hh h_{t-1} + b)   [via nn.RNN, size rec_dim]
        lin  = lin_act(W_lin x_t + b_lin)               [size n_out]
        y_t  = W_rec_proj(h_t) + lin                    [size n_out]

    W_rec_proj projects the recurrent hidden state to n_out before summing.
    The hidden state carries only h_t (shape rec_dim).

    recurrent_kwargs keys
    ----------------------
    rec_dim  : hidden size of the RNN branch    (default: n_out)
    lin_act  : activation for the linear branch (default: torch.relu)
    """

    _ACT_TO_NONLINEARITY = {torch.tanh: "tanh", torch.relu: "relu"}

    def __init__(
        self,
        n_in,
        n_out,
        rec_dim=None,
        rec_act=torch.tanh,
        lin_act=torch.relu,
    ):
        super().__init__()
        rec_dim = rec_dim if rec_dim is not None else n_out
        self.n_in = n_in
        self.n_out = n_out
        self.rec_dim = rec_dim
        self.rec_act = rec_act
        self.lin_act = lin_act

        nonlinearity = self._ACT_TO_NONLINEARITY.get(rec_act)
        if nonlinearity is None:
            raise ValueError(
                f"ParallelRecurrentCell recurrent branch only supports torch.tanh or torch.relu; got {rec_act}."
            )
        self.rnn = torch.nn.RNN(input_size=n_in, hidden_size=rec_dim, batch_first=False, nonlinearity=nonlinearity)
        self.w_rec_proj = torch.nn.Linear(rec_dim, n_out)
        self.w_lin = torch.nn.Linear(n_in, n_out)
        self.state = None

        torch.nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        torch.nn.init.orthogonal_(self.rnn.weight_hh_l0)
        torch.nn.init.zeros_(self.rnn.bias_ih_l0)
        torch.nn.init.zeros_(self.rnn.bias_hh_l0)
        torch.nn.init.xavier_uniform_(self.w_rec_proj.weight)
        torch.nn.init.zeros_(self.w_rec_proj.bias)
        torch.nn.init.xavier_uniform_(self.w_lin.weight)
        torch.nn.init.zeros_(self.w_lin.bias)

    def reset(self, batch_size: int = 1, device=None, dtype=None):
        device = device if device is not None else self.rnn.weight_hh_l0.device
        dtype = dtype if dtype is not None else self.rnn.weight_hh_l0.dtype
        self.state = torch.zeros(batch_size, self.rec_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step forward.  x: (B, n_in)  →  (B, n_out)"""
        B = x.shape[0]
        if self.state is None or self.state.shape[0] != B:
            self.reset(B, device=x.device, dtype=x.dtype)
        _, h_n = self.rnn(x.unsqueeze(0), self.state.unsqueeze(0))
        self.state = h_n.squeeze(0)
        return self.w_rec_proj(self.state) + self.lin_act(self.w_lin(x))

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward.  x_seq: (T, n_in)  →  (T, n_out)"""
        x_seq_3d, squeeze_batch = _normalize_sequence_input(x_seq, expected_in=self.n_in)
        B = x_seq_3d.shape[1]
        if self.state is None or self.state.shape[0] != B:
            self.reset(B, device=x_seq.device, dtype=x_seq.dtype)
        out, h_n = self.rnn(x_seq_3d, self.state.unsqueeze(0))              # (T, B, rec_dim)
        self.state = h_n.squeeze(0)
        y = self.w_rec_proj(out) + self.lin_act(self.w_lin(x_seq_3d))       # (T, B, n_out)
        return y.squeeze(1) if squeeze_batch else y

    def get_state(self, clone: bool = True, detach: bool = True) -> torch.Tensor:
        if self.state is None:
            return torch.empty(0, device=self.rnn.weight_hh_l0.device, dtype=self.rnn.weight_hh_l0.dtype)
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
        self.state = s.to(device=self.rnn.weight_hh_l0.device, dtype=self.rnn.weight_hh_l0.dtype)


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
    if mode in ("oja", "oja_grad", "gradient_oja", "plastic"):
        kwargs = dict(recurrent_kwargs or {})
        eta = float(kwargs.pop("eta", 1e-3))
        lambda_ = float(kwargs.pop("lambda_", kwargs.pop("lambda", 1e-4)))
        alpha = float(kwargs.pop("alpha", 1.0))
        nu = float(kwargs.pop("nu", 0.1))
        wclip = kwargs.pop("wclip", None)
        act = kwargs.pop("act", activation)
        if kwargs:
            invalid = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported Oja recurrent_kwargs: {invalid}.")
        return GradientOjaRecurrentCell(n_in, n_out, act=act, eta=eta, lambda_=lambda_, alpha=alpha, nu=nu, wclip=wclip)
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
        return LMUCell(n_in, n_out, memory_size=memory_size, theta=theta, act=act, learn_a_b=learn_a_b)
    if mode in ("parallel", "parallel_rnn", "parallel_recurrent"):
        kwargs = dict(recurrent_kwargs or {})
        rec_dim = kwargs.pop("rec_dim", None)
        lin_act = kwargs.pop("lin_act", torch.relu)
        if kwargs:
            invalid = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported parallel recurrent_kwargs: {invalid}.")
        return ParallelRecurrentCell(n_in, n_out, rec_dim=rec_dim, rec_act=activation, lin_act=lin_act)
    raise ValueError(f"Unsupported recurrent_type '{recurrent_type}'. Use 'simple', 'oja', 'gru', 'lmu', or 'parallel'.")


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

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Process a full sequence in one call.

        x_seq: (T, n_in)  →  (T, n_out)

        For GRU cells this dispatches to forward_seq() which runs as a single
        CuDNN kernel.  For all other cell types it loops internally, which still
        avoids the Python wrapper overhead of the old outer loop.

        nn.Linear supports arbitrary leading dimensions, so self.out(z) on a
        (T, prev_dim) tensor produces (T, n_out) correctly with no extra code.
        """
        z = x_seq  # (T, n_in)
        for cell in self.cells:
            z = cell.forward_seq(z)  # (T, hidden_dim)
        return self.out_act(self.out(z))  # (T, n_out)

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

    def plastic_modules(self) -> list[torch.nn.Module]:
        return _collect_plastic_modules(self.cells)

    def plastic_parameters(self) -> list[torch.nn.Parameter]:
        params = []
        for cell in self.plastic_modules():
            params.extend(cell.plastic_parameters())
        return params

    def get_online_traces(self, clone: bool = True, detach: bool = True) -> list[dict[str, torch.Tensor] | None]:
        traces = []
        for cell in self.cells:
            if hasattr(cell, "get_online_trace"):
                traces.append(cell.get_online_trace(clone=clone, detach=detach))
            else:
                traces.append(None)
        return traces

    def set_online_traces(
        self,
        traces: list[dict[str, torch.Tensor] | None],
        clone: bool = True,
        detach: bool = True,
        strict: bool = True,
    ):
        traces = traces or []
        if strict and len(traces) != len(self.cells):
            raise ValueError(f"Expected {len(self.cells)} cell traces, got {len(traces)}.")
        for cell, trace in zip(self.cells, traces):
            if hasattr(cell, "set_online_trace"):
                cell.set_online_trace(trace, clone=clone, detach=detach)

    def online_update(self, loss: torch.Tensor, *, retain_graph: bool = False) -> bool:
        return _apply_online_updates(self.cells, loss, retain_graph=retain_graph)


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
        self._trunk_out_dim = prev

        self.actor = RecurrentNetwork(prev, action_dim, hidden_dims=hidden_dims_actor, activation=activation, recurrent_type=recurrent_type, recurrent_kwargs=self.recurrent_kwargs)
        self.value = RecurrentNetwork(prev, 1, hidden_dims=hidden_dims_value, activation=activation, recurrent_type=recurrent_type, recurrent_kwargs=self.recurrent_kwargs)

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
            z = layer(z)
        return z

    def _forward_seq_trunk(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (T, state_dim)  →  (T, trunk_out_dim)

        Runs each trunk cell's forward_seq (CuDNN path for GRU) and applies
        the same outer activation as the single-step forward().
        """
        z = x_seq
        for cell in self.layers:
            z = cell.forward_seq(z)  # (T, hidden_dim)
        return z

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (T, state_dim)  →  (T, trunk_out_dim)
        Alias kept for symmetry; heads call their own forward_seq separately.
        """
        return self._forward_seq_trunk(x_seq)

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

    def plastic_modules(self) -> list[torch.nn.Module]:
        return _collect_plastic_modules(self.layers) + self.actor.plastic_modules() + self.value.plastic_modules()

    def plastic_parameters(self) -> list[torch.nn.Parameter]:
        params = []
        for module in self.plastic_modules():
            params.extend(module.plastic_parameters())
        return params

    def get_online_traces(self, clone: bool = True, detach: bool = True) -> dict:
        trunk = []
        for cell in self.layers:
            if hasattr(cell, "get_online_trace"):
                trunk.append(cell.get_online_trace(clone=clone, detach=detach))
            else:
                trunk.append(None)
        return {"trunk": trunk, "actor": self.actor.get_online_traces(clone=clone, detach=detach), "value": self.value.get_online_traces(clone=clone, detach=detach)}

    def set_online_traces(self, traces: dict, clone: bool = True, detach: bool = True, strict: bool = True):
        traces = traces or {}
        if strict:
            for key in ("trunk", "actor", "value"):
                if key not in traces:
                    raise KeyError(f"Missing key '{key}' in online trace snapshot.")
        trunk_traces = traces.get("trunk", [])
        actor_traces = traces.get("actor", [])
        value_traces = traces.get("value", [])
        if strict and len(trunk_traces) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} trunk traces, got {len(trunk_traces)}.")
        for cell, trace in zip(self.layers, trunk_traces):
            if hasattr(cell, "set_online_trace"):
                cell.set_online_trace(trace, clone=clone, detach=detach)
        self.actor.set_online_traces(actor_traces, clone=clone, detach=detach, strict=strict)
        self.value.set_online_traces(value_traces, clone=clone, detach=detach, strict=strict)

    def online_update(self, loss: torch.Tensor, *, retain_graph: bool = False) -> bool:
        return _apply_online_updates(self.plastic_modules(), loss, retain_graph=retain_graph)


class RecurrentActionQNetwork(torch.nn.Module):
    """
    Shared recurrent trunk → actor head (logits) + Q head (per-action Q-values).

    forward(x)         →  (logits, q_values)   single step, as before
    forward_seq(x_seq) →  (logits_seq, q_seq)  full sequence, CuDNN path for GRU
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=None,
        hidden_dims_actor=None,
        hidden_dims_q=None,
        activation=torch.relu,
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
        self._trunk_out_dim = prev

        self.actor = RecurrentNetwork(prev, action_dim, hidden_dims=hidden_dims_actor, activation=activation, recurrent_type=recurrent_type, recurrent_kwargs=self.recurrent_kwargs)
        self.q     = RecurrentNetwork(prev, action_dim, hidden_dims=hidden_dims_q,     activation=activation, recurrent_type=recurrent_type, recurrent_kwargs=self.recurrent_kwargs)

    def reset(self, batch_size: int = 1):
        dev = next(self.parameters()).device
        dt  = next(self.parameters()).dtype
        for cell in self.layers:
            cell.reset(batch_size, device=dev, dtype=dt)
        self.actor.reset(batch_size, device=dev, dtype=dt)
        self.q.reset(batch_size, device=dev, dtype=dt)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step trunk forward.  x: (B, state_dim) → (B, trunk_out_dim)"""
        z = x.view(-1, self.state_dim)
        for layer in self.layers:
            z = layer(z)
        return z

    def _trunk_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Full-sequence trunk forward.  x_seq: (T, state_dim) → (T, trunk_out_dim)

        Uses each cell's forward_seq(); for GRU this is the CuDNN fused kernel.
        """
        z = x_seq
        for cell in self.layers:
            z = cell.forward_seq(z)
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward.  Returns (logits, q_values), each (B, action_dim)."""
        z = self._trunk(x)
        return self.actor(z), self.q(z)

    def forward_seq(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full-sequence forward.  x_seq: (T, state_dim)

        Returns:
          logits_seq : (T, action_dim)
          q_seq      : (T, action_dim)

        The trunk runs via CuDNN for GRU; the actor and Q heads run their own
        forward_seq() so their recurrent cells also benefit from the same path.
        nn.Linear handles (T, dim) inputs natively — no extra looping needed.
        """
        z = self._trunk_seq(x_seq)              # (T, trunk_out_dim)
        logits_seq = self.actor.forward_seq(z)  # (T, action_dim)
        q_seq      = self.q.forward_seq(z)      # (T, action_dim)
        return logits_seq, q_seq

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        trunk = [cell.get_state(clone=clone, detach=detach) for cell in self.layers]
        return {"trunk": trunk, "actor": self.actor.get_states(clone=clone, detach=detach), "q": self.q.get_states(clone=clone, detach=detach)}

    def set_states(self, states: dict, clone: bool = True, detach: bool = True, strict: bool = True):
        if strict:
            for k in ("trunk", "actor", "q"):
                if k not in states:
                    raise KeyError(f"Missing key '{k}' in states snapshot.")
        trunk_states = states.get("trunk", [])
        if strict and len(trunk_states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} trunk states, got {len(trunk_states)}.")
        for cell, s in zip(self.layers, trunk_states):
            cell.set_state(s, clone=clone, detach=detach)
        self.actor.set_states(states.get("actor", []), clone=clone, detach=detach, strict=strict)
        self.q.set_states(states.get("q", []), clone=clone, detach=detach, strict=strict)

    def plastic_modules(self) -> list[torch.nn.Module]:
        return _collect_plastic_modules(self.layers) + self.actor.plastic_modules() + self.q.plastic_modules()

    def plastic_parameters(self) -> list[torch.nn.Parameter]:
        params = []
        for module in self.plastic_modules():
            params.extend(module.plastic_parameters())
        return params

    def online_update(self, loss: torch.Tensor, *, retain_graph: bool = False) -> bool:
        return _apply_online_updates(self.plastic_modules(), loss, retain_graph=retain_graph)
