import torch

from src.network.core.per_asset import BaseFlatPerAssetActionValueNetwork
from src.network.core.vanilla import VanillaNetwork


class AuxiliaryPerAssetActionValueNetwork(BaseFlatPerAssetActionValueNetwork):
    """
    ``FlatPerAssetActionValueNetwork`` plus supervised predictive heads.

    Public inference returns ``(logits, critic, aux)`` where ``critic`` is a
    scalar value by default or per-action Q-values when ``critic_type="q"``,
    and ``aux`` contains:
        reward    : (...,)
        asset_ret : (..., num_assets, aux_horizons)
        asset_vol : (..., num_assets, aux_horizons)
    """

    name = "auxiliary_per_asset_action_value"

    def __init__(
        self,
        *args,
        aux_horizons=(1, 5, 20),
        hidden_dims_aux=None,
        critic_type: str = "v",
        **kwargs,
    ):
        critic_type = str(critic_type).lower()
        if critic_type not in ("v", "q"):
            raise ValueError(f"critic_type must be 'v' or 'q', got '{critic_type}'.")
        hidden_dims_value = kwargs.get("hidden_dims_value", None)
        super().__init__(*args, **kwargs)
        hidden_dims_aux = hidden_dims_aux or []
        hidden_dims_value = hidden_dims_value or []
        self.critic_type = critic_type
        self.aux_horizons = tuple(int(h) for h in aux_horizons)
        if len(self.aux_horizons) < 1:
            raise ValueError("aux_horizons must contain at least one horizon.")
        if any(h <= 0 for h in self.aux_horizons):
            raise ValueError(f"aux_horizons must be positive, got {self.aux_horizons}.")
        self.n_aux_horizons = len(self.aux_horizons)

        head_in = self.d_mem if self.combine_mode == "add" else self.d_mem + self.d_ff
        if self.critic_type == "q":
            self.value = VanillaNetwork(
                head_in,
                self.action_dim,
                hidden_dims=hidden_dims_value,
                activation=self.activation,
            )
        self.aux_reward = VanillaNetwork(head_in, 1, hidden_dims=hidden_dims_aux, activation=self.activation)
        self.aux_asset_ret = VanillaNetwork(
            head_in,
            self.num_assets * self.n_aux_horizons,
            hidden_dims=hidden_dims_aux,
            activation=self.activation,
        )
        self.aux_asset_vol = VanillaNetwork(
            head_in,
            self.num_assets * self.n_aux_horizons,
            hidden_dims=hidden_dims_aux,
            activation=self.activation,
        )

    def _build_aux(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "reward": self.aux_reward(h).squeeze(-1),
            "asset_ret": self.aux_asset_ret(h).view(h.shape[0], self.num_assets, self.n_aux_horizons),
            "asset_vol": torch.nn.functional.softplus(
                self.aux_asset_vol(h).view(h.shape[0], self.num_assets, self.n_aux_horizons)
            ),
        }

    def _critic(self, h: torch.Tensor, prefix_shape: torch.Size) -> torch.Tensor:
        critic = self.value(h)
        if self.critic_type == "q":
            return critic.reshape(prefix_shape + (self.action_dim,))
        return critic.squeeze(-1).reshape(prefix_shape)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        shape = x.shape
        x_flat = x.reshape(-1, self.state_dim)
        z = self._encode_flat(x_flat)
        h_mem = self.recurrent(z)
        h_ff = self.feedforward(z)
        h = self._combine(h_mem, h_ff)

        logits = self.actor(h).reshape(shape[:-1] + (self.action_dim,))
        values = self._critic(h, shape[:-1])
        aux_flat = self._build_aux(h)
        aux = {
            "reward": aux_flat["reward"].reshape(shape[:-1]),
            "asset_ret": aux_flat["asset_ret"].reshape(shape[:-1] + (self.num_assets, self.n_aux_horizons)),
            "asset_vol": aux_flat["asset_vol"].reshape(shape[:-1] + (self.num_assets, self.n_aux_horizons)),
        }
        return logits, values, aux

    def forward_seq(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if x_seq.dim() == 2:
            if x_seq.shape[-1] != self.state_dim:
                raise ValueError(f"Expected last dim {self.state_dim}, got {x_seq.shape[-1]}.")
            T = x_seq.shape[0]
            B = 1
            x_batched = x_seq.unsqueeze(1)
            squeeze_batch = True
        elif x_seq.dim() == 3:
            if x_seq.shape[-1] != self.state_dim:
                raise ValueError(f"Expected last dim {self.state_dim}, got {x_seq.shape[-1]}.")
            T, B, _ = x_seq.shape
            x_batched = x_seq
            squeeze_batch = False
        else:
            raise ValueError(
                f"Expected sequence input with shape (T, {self.state_dim}) or "
                f"(T, B, {self.state_dim}), got {tuple(x_seq.shape)}."
            )

        x_flat = x_batched.reshape(T * B, self.state_dim)
        z_flat = self._encode_flat(x_flat)
        z_seq = z_flat.view(T, B, -1)

        h_mem_seq = self.recurrent.forward_seq(z_seq)
        h_ff_seq = self.feedforward(z_flat).view(T, B, self.d_ff)
        h_seq = self._combine(h_mem_seq, h_ff_seq)
        h_flat = h_seq.reshape(T * B, -1)

        logits_seq = self.actor(h_flat).view(T, B, self.action_dim)
        if self.critic_type == "q":
            values_seq = self.value(h_flat).view(T, B, self.action_dim)
        else:
            values_seq = self.value(h_flat).squeeze(-1).view(T, B)
        aux_flat = self._build_aux(h_flat)
        aux_seq = {
            "reward": aux_flat["reward"].view(T, B),
            "asset_ret": aux_flat["asset_ret"].view(T, B, self.num_assets, self.n_aux_horizons),
            "asset_vol": aux_flat["asset_vol"].view(T, B, self.num_assets, self.n_aux_horizons),
        }

        if squeeze_batch:
            logits_seq = logits_seq.squeeze(1)
            values_seq = values_seq.squeeze(1)
            aux_seq = {key: value.squeeze(1) for key, value in aux_seq.items()}
        return logits_seq, values_seq, aux_seq
