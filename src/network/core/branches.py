import torch

from src.network.core.recurrent import RecurrentNetwork, _build_recurrent_cell
from src.network.core.vanilla import VanillaNetwork


class LatentRecurrentFeedForwardBranch(torch.nn.Module):
    """Parallel recurrent/feedforward branch used by latent model networks."""

    def __init__(
        self,
        n_in: int,
        d_mem: int,
        d_ff: int,
        hidden_dims_mem=None,
        hidden_dims_ff=None,
        combine_mode: str = "concat",
        activation=torch.tanh,
        recurrent_activation=torch.tanh,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
    ):
        super().__init__()
        if combine_mode not in ("concat", "add"):
            raise ValueError(f"combine_mode must be 'concat' or 'add', got '{combine_mode}'.")
        if combine_mode == "add" and int(d_mem) != int(d_ff):
            raise ValueError(
                f"combine_mode='add' requires d_mem == d_ff, got d_mem={d_mem}, d_ff={d_ff}."
            )
        hidden_dims_mem = hidden_dims_mem or []
        hidden_dims_ff = hidden_dims_ff or []
        recurrent_kwargs = dict(recurrent_kwargs or {})

        self.n_in = int(n_in)
        self.d_mem = int(d_mem)
        self.d_ff = int(d_ff)
        self.combine_mode = combine_mode
        self.out_dim = self.d_mem if self.combine_mode == "add" else self.d_mem + self.d_ff

        if hidden_dims_mem:
            self.recurrent = RecurrentNetwork(
                self.n_in,
                self.d_mem,
                hidden_dims=hidden_dims_mem,
                activation=recurrent_activation,
                recurrent_type=recurrent_type,
                recurrent_kwargs=recurrent_kwargs,
            )
        else:
            self.recurrent = _build_recurrent_cell(
                self.n_in,
                self.d_mem,
                recurrent_activation,
                recurrent_type,
                recurrent_kwargs,
            )
        self._recurrent_is_network = isinstance(self.recurrent, RecurrentNetwork)
        self.feedforward = VanillaNetwork(
            self.n_in,
            self.d_ff,
            hidden_dims=hidden_dims_ff,
            activation=activation,
        )

    def _combine(self, h_mem: torch.Tensor, h_ff: torch.Tensor) -> torch.Tensor:
        if self.combine_mode == "add":
            return h_mem + h_ff
        return torch.cat([h_mem, h_ff], dim=-1)

    def reset(self, batch_size: int = 1, device: torch.device = None, dtype: torch.dtype = None):
        self.recurrent.reset(batch_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._combine(self.recurrent(x), self.feedforward(x))

    def forward_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        shape = x_seq.shape
        h_mem = self.recurrent.forward_seq(x_seq)
        h_ff = self.feedforward(x_seq.reshape(-1, self.n_in)).view(*shape[:-1], self.d_ff)
        return self._combine(h_mem, h_ff)

    def get_state(self, clone: bool = True, detach: bool = True):
        if self._recurrent_is_network:
            return self.recurrent.get_states(clone=clone, detach=detach)
        return self.recurrent.get_state(clone=clone, detach=detach)

    def set_state(self, state, clone: bool = True, detach: bool = True, strict: bool = True):
        if self._recurrent_is_network:
            self.recurrent.set_states(state, clone=clone, detach=detach, strict=strict)
        else:
            self.recurrent.set_state(state, clone=clone, detach=detach)
