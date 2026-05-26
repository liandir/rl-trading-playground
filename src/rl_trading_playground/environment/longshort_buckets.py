r"""Pure-discrete long/short multi-currency environment with size buckets.

Subclasses [rl_trading_playground.environment.generic.longshort](../generic/longshort.html)
to replace its continuous fraction $a^{\mathrm c} \in [0, 1]$ with a
fixed set of $K$ size buckets
$\mathbf{q} = (q_1, \dots, q_K) \in (0, 1]^K$ (default
$\mathbf{q} = (0.5, 1.0)$). State features are inherited unchanged.

State space
===========

Same flat layout as
[rl_trading_playground.environment.generic.longshort](../generic/longshort.html):

$$
\dim(\mathcal{S}) = 4 M N + 9 N + 8.
$$

Action space
============

Single flat discrete action $a \in \{0, 1, \dots, 3NK\}$:

$$
a =
\begin{cases}
0 & \text{hold},\\
1 \dots NK & \text{open LONG with bucket } q_{j_a} \text{ on asset } k_a,\\
NK+1 \dots 2NK & \text{open SHORT with bucket } q_{j_a} \text{ on asset } k_a,\\
2NK+1 \dots 3NK & \text{CLOSE bucket } q_{j_a} \text{ of asset } k_a.
\end{cases}
$$

For each non-hold action

$$
(k_a, j_a) =
\big(\,\lfloor (a-1 \bmod NK) / K \rfloor,\;
(a-1) \bmod K\,\big).
$$

Open commits $q_{j_a} \cdot C_t$ of available cash as collateral on
$k_a$. Flips auto-close the full opposite position first. Close
reduces position size by $q_{j_a} \cdot |u_{k_a}|$ units. Same-side
re-opens are rejected (validity exposed via ``valid_action_mask``).

Reward and termination match
[rl_trading_playground.environment.generic.longshort](../generic/longshort.html).
"""
from typing import Tuple

import torch

from rl_trading_playground.environment.generic.longshort import (
    BatchedMultiCurrencyEnv,
    MultiCurrencyEnv,
    State,
    StateHistory,
)


class LongShortEnv(MultiCurrencyEnv):
    """
    Discrete bucketized long/short trading environment.

    This is the discrete counterpart to rl_trading_playground.environment.hybrid.longshort:
    the continuous fraction is replaced by predefined size buckets.

    Action space:
        0                                   -> hold
        1 .. N*K                            -> open LONG  asset i with bucket k
        N*K+1 .. 2*N*K                      -> open SHORT asset i with bucket k
        2*N*K+1 .. 3*N*K                    -> CLOSE      asset i with bucket k

    where:
        K = len(size_buckets)
        size_buckets default to (0.50, 1.00)

    Open semantics:
        commit bucket_fraction * available_cash as collateral on the target
        asset. Flips auto-close the full opposite position first.

    Close semantics:
        close bucket_fraction * current position units on the target asset.

    Same-side opens are rejected to preserve the current longshort behavior.
    """

    def __init__(
        self,
        N: int,
        C0: float,
        tau_p: torch.Tensor,
        *,
        bankruptcy_threshold: float = 1.0,
        transaction_eps: float = 1e-2,
        use_dollar_volume: bool = True,
        save_history: bool = False,
        open_fee: float = 1.0,
        close_fee: float = 1.0,
        tax_rate: float = 0.26,
        min_open_dollars: float = 10.0,
        val_coeff: float = 1.0,
        roi_coeff: float = 1.0,
        reward_mode: str = "log",
        size_buckets: Tuple[float, ...] = (0.50, 1.00),
        done_reward_penalty: float = 1.0,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the instance.

        Args:
            N (int): The n value.
            C0 (float): The c0 value.
            tau_p (torch.Tensor): The tau p value.
            bankruptcy_threshold (float): The bankruptcy threshold value. Defaults to ``1.0``.
            transaction_eps (float): The transaction eps value. Defaults to ``0.01``.
            use_dollar_volume (bool): The use dollar volume value. Defaults to ``True``.
            save_history (bool): The save history value. Defaults to ``False``.
            open_fee (float): The open fee value. Defaults to ``1.0``.
            close_fee (float): The close fee value. Defaults to ``1.0``.
            tax_rate (float): The tax rate value. Defaults to ``0.26``.
            min_open_dollars (float): The min open dollars value. Defaults to ``10.0``.
            val_coeff (float): The val coeff value. Defaults to ``1.0``.
            roi_coeff (float): The roi coeff value. Defaults to ``1.0``.
            reward_mode (str): The reward mode value. Defaults to ``'log'``.
            size_buckets (Tuple[float, ...]): The size buckets value. Defaults to ``(0.5, 1.0)``.
            done_reward_penalty (float): The done reward penalty value. Defaults to ``1.0``.
            dtype (torch.dtype): The dtype value. Defaults to ``torch.float32``.
            eps (float): The eps value. Defaults to ``1e-08``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(
            N=N,
            C0=C0,
            tau_p=tau_p,
            bankruptcy_threshold=bankruptcy_threshold,
            transaction_eps=transaction_eps,
            use_dollar_volume=use_dollar_volume,
            save_history=save_history,
            open_fee=open_fee,
            close_fee=close_fee,
            tax_rate=tax_rate,
            min_open_dollars=min_open_dollars,
            val_coeff=val_coeff,
            roi_coeff=roi_coeff,
            reward_mode=reward_mode,
            done_reward_penalty=done_reward_penalty,
            dtype=dtype,
            eps=eps,
        )

        assert len(size_buckets) > 0, "size_buckets must not be empty"
        assert all(0.0 < x <= 1.0 for x in size_buckets), "all size buckets must be in (0, 1]"

        self.size_buckets = torch.tensor(size_buckets, dtype=self.dtype, device=self.tau_p.device)
        self.K = int(self.size_buckets.numel())
        self.action_dim = 1 + 3 * self.N * self.K

    # ------------------------------------------------------------------
    # action helpers
    # ------------------------------------------------------------------

    def encode_hold(self) -> int:
        """Encode hold for LongShortEnv.

        Returns:
            int: The computed or requested result.
        """
        return 0

    def encode_long(self, asset_idx: int, bucket_idx: int) -> int:
        """Encode long for LongShortEnv.

        Args:
            asset_idx (int): The asset idx value.
            bucket_idx (int): The bucket idx value.

        Returns:
            int: The computed or requested result.
        """
        if not (0 <= asset_idx < self.N):
            raise ValueError("asset_idx out of range")
        if not (0 <= bucket_idx < self.K):
            raise ValueError("bucket_idx out of range")
        return 1 + asset_idx * self.K + bucket_idx

    def encode_short(self, asset_idx: int, bucket_idx: int) -> int:
        """Encode short for LongShortEnv.

        Args:
            asset_idx (int): The asset idx value.
            bucket_idx (int): The bucket idx value.

        Returns:
            int: The computed or requested result.
        """
        if not (0 <= asset_idx < self.N):
            raise ValueError("asset_idx out of range")
        if not (0 <= bucket_idx < self.K):
            raise ValueError("bucket_idx out of range")
        return 1 + self.N * self.K + asset_idx * self.K + bucket_idx

    def encode_close(self, asset_idx: int, bucket_idx: int) -> int:
        """Encode close for LongShortEnv.

        Args:
            asset_idx (int): The asset idx value.
            bucket_idx (int): The bucket idx value.

        Returns:
            int: The computed or requested result.
        """
        if not (0 <= asset_idx < self.N):
            raise ValueError("asset_idx out of range")
        if not (0 <= bucket_idx < self.K):
            raise ValueError("bucket_idx out of range")
        return 1 + 2 * self.N * self.K + asset_idx * self.K + bucket_idx

    def decode_action(self, action_idx: int) -> tuple[str, int | None, float | None]:
        """Decode action for LongShortEnv.

        Args:
            action_idx (int): The action idx value.

        Returns:
            tuple[str, int | None, float | None]: The computed or requested result.
        """
        if action_idx == 0:
            return ("hold", None, None)

        block = self.N * self.K
        short_offset = 1 + block
        close_offset = 1 + 2 * block

        if 1 <= action_idx < short_offset:
            z = action_idx - 1
            asset_idx = z // self.K
            bucket_idx = z % self.K
            return ("long", asset_idx, float(self.size_buckets[bucket_idx].item()))

        if short_offset <= action_idx < close_offset:
            z = action_idx - short_offset
            asset_idx = z // self.K
            bucket_idx = z % self.K
            return ("short", asset_idx, float(self.size_buckets[bucket_idx].item()))

        if close_offset <= action_idx < self.action_dim:
            z = action_idx - close_offset
            asset_idx = z // self.K
            bucket_idx = z % self.K
            return ("close", asset_idx, float(self.size_buckets[bucket_idx].item()))

        raise ValueError(f"invalid action index {action_idx}")

    def valid_action_mask(self) -> torch.Tensor:
        """Valid action mask for LongShortEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self._state_device())
        mask[0] = True

        cash = float(self.C.item())

        for k in range(self.N):
            units = float(self.pos_units[k].item())
            is_long = units > self.eps
            is_short = units < -self.eps
            is_flat = not (is_long or is_short)

            price = float(self.p[k].item())
            notional_full = abs(units) * price
            close_ok_full = (not is_flat) and notional_full > self.c_fee

            if is_flat:
                cash_avail = cash
            elif close_ok_full:
                committed_k = float(self.committed[k].item())
                entry = float(self.entry_price[k].item())
                pnl = units * (price - entry)
                gross = committed_k + pnl
                proceeds = max(gross - self.c_fee, 0.0)
                pre_tax = proceeds - committed_k
                tax = max(pre_tax, 0.0) * self.tax_rate
                net_proceeds = proceeds - tax
                cash_avail = cash + net_proceeds
            else:
                cash_avail = cash

            for b_idx in range(self.K):
                frac = float(self.size_buckets[b_idx].item())
                budget = frac * cash_avail
                can_open = budget > self.o_fee and (budget - self.o_fee) >= self.min_open_dollars

                if is_long:
                    mask[self.encode_long(k, b_idx)] = False
                elif is_short:
                    mask[self.encode_long(k, b_idx)] = close_ok_full and can_open
                else:
                    mask[self.encode_long(k, b_idx)] = can_open

                if is_short:
                    mask[self.encode_short(k, b_idx)] = False
                elif is_long:
                    mask[self.encode_short(k, b_idx)] = close_ok_full and can_open
                else:
                    mask[self.encode_short(k, b_idx)] = can_open

                close_notional = frac * notional_full
                mask[self.encode_close(k, b_idx)] = (not is_flat) and close_notional > self.c_fee

        return mask

    # ------------------------------------------------------------------
    # trading
    # ------------------------------------------------------------------

    def _trade(self, a) -> tuple[torch.Tensor, bool]:
        """Trade for LongShortEnv.

        Args:
            a (Any): The a value.

        Returns:
            tuple[torch.Tensor, bool]: The computed or requested result.
        """
        self.realized_cost.zero_()
        self.realized_pnl.zero_()

        if a is None:
            action_idx = 0
        elif isinstance(a, torch.Tensor):
            action_idx = int(a.item())
        else:
            action_idx = int(a)

        if not (0 <= action_idx < self.action_dim):
            raise ValueError(f"Invalid discrete action {action_idx}, expected in [0, {self.action_dim - 1}]")

        action_info = torch.tensor([0.0, -1.0, 0.0], dtype=self.dtype, device=self._state_device())

        if action_idx == 0:
            return action_info, True

        kind, k, frac = self.decode_action(action_idx)
        action_info[1] = float(k)
        action_info[2] = float(frac)

        if kind == "close":
            action_info[0] = 3.0
            did = self._close_position(k, frac)
            return action_info, did

        side = 1 if kind == "long" else -1
        action_info[0] = 1.0 if side == 1 else 2.0

        units = float(self.pos_units[k].item())
        if (side == 1 and units < -self.eps) or (side == -1 and units > self.eps):
            self._close_position(k, frac=1.0)

        units_after = float(self.pos_units[k].item())
        if (side == 1 and units_after > self.eps) or (side == -1 and units_after < -self.eps):
            return action_info, False

        did = self._open_position(k, side, frac)

        dust = self.committed < self.transaction_eps
        self.pos_units[dust] = 0.0
        self.committed[dust] = 0.0
        self.entry_price[dust] = 0.0
        self.C = torch.clamp(self.C, min=0.0)

        return action_info, did


class BatchedLongShortEnv(BatchedMultiCurrencyEnv):
    """
    Fully batched bucketized long/short environment.

    This reuses the hybrid longshort execution logic after translating each
    bucketized discrete action into:
      - the base long/short/close action on one asset
      - the selected bucket fraction
    """

    def __init__(
        self,
        B: int,
        N: int,
        C0: float,
        tau_p: torch.Tensor,
        *,
        bankruptcy_threshold: float = 1.0,
        transaction_eps: float = 1e-2,
        use_dollar_volume: bool = True,
        open_fee: float = 1.0,
        close_fee: float = 1.0,
        tax_rate: float = 0.26,
        min_open_dollars: float = 10.0,
        val_coeff: float = 1.0,
        roi_coeff: float = 1.0,
        reward_mode: str = "log",
        size_buckets: Tuple[float, ...] = (0.50, 1.00),
        done_reward_penalty: float = 1.0,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the instance.

        Args:
            B (int): The b value.
            N (int): The n value.
            C0 (float): The c0 value.
            tau_p (torch.Tensor): The tau p value.
            bankruptcy_threshold (float): The bankruptcy threshold value. Defaults to ``1.0``.
            transaction_eps (float): The transaction eps value. Defaults to ``0.01``.
            use_dollar_volume (bool): The use dollar volume value. Defaults to ``True``.
            open_fee (float): The open fee value. Defaults to ``1.0``.
            close_fee (float): The close fee value. Defaults to ``1.0``.
            tax_rate (float): The tax rate value. Defaults to ``0.26``.
            min_open_dollars (float): The min open dollars value. Defaults to ``10.0``.
            val_coeff (float): The val coeff value. Defaults to ``1.0``.
            roi_coeff (float): The roi coeff value. Defaults to ``1.0``.
            reward_mode (str): The reward mode value. Defaults to ``'log'``.
            size_buckets (Tuple[float, ...]): The size buckets value. Defaults to ``(0.5, 1.0)``.
            done_reward_penalty (float): The done reward penalty value. Defaults to ``1.0``.
            dtype (torch.dtype): The dtype value. Defaults to ``torch.float32``.
            eps (float): The eps value. Defaults to ``1e-08``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(
            B=B,
            N=N,
            C0=C0,
            tau_p=tau_p,
            bankruptcy_threshold=bankruptcy_threshold,
            transaction_eps=transaction_eps,
            use_dollar_volume=use_dollar_volume,
            open_fee=open_fee,
            close_fee=close_fee,
            tax_rate=tax_rate,
            min_open_dollars=min_open_dollars,
            val_coeff=val_coeff,
            roi_coeff=roi_coeff,
            reward_mode=reward_mode,
            done_reward_penalty=done_reward_penalty,
            dtype=dtype,
            eps=eps,
        )

        assert len(size_buckets) > 0, "size_buckets must not be empty"
        assert all(0.0 < x <= 1.0 for x in size_buckets), "all size buckets must be in (0, 1]"

        self.size_buckets = torch.tensor(size_buckets, dtype=self.dtype, device=self.tau_p.device)
        self.K = int(self.size_buckets.numel())
        self.action_dim = 1 + 3 * self.N * self.K

    # ------------------------------------------------------------------
    # action helpers
    # ------------------------------------------------------------------

    def encode_hold(self) -> int:
        """Encode hold for BatchedLongShortEnv.

        Returns:
            int: The computed or requested result.
        """
        return 0

    def encode_long(self, asset_idx: int, bucket_idx: int) -> int:
        """Encode long for BatchedLongShortEnv.

        Args:
            asset_idx (int): The asset idx value.
            bucket_idx (int): The bucket idx value.

        Returns:
            int: The computed or requested result.
        """
        return 1 + asset_idx * self.K + bucket_idx

    def encode_short(self, asset_idx: int, bucket_idx: int) -> int:
        """Encode short for BatchedLongShortEnv.

        Args:
            asset_idx (int): The asset idx value.
            bucket_idx (int): The bucket idx value.

        Returns:
            int: The computed or requested result.
        """
        return 1 + self.N * self.K + asset_idx * self.K + bucket_idx

    def encode_close(self, asset_idx: int, bucket_idx: int) -> int:
        """Encode close for BatchedLongShortEnv.

        Args:
            asset_idx (int): The asset idx value.
            bucket_idx (int): The bucket idx value.

        Returns:
            int: The computed or requested result.
        """
        return 1 + 2 * self.N * self.K + asset_idx * self.K + bucket_idx

    def _split_bucket_actions(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split bucket actions for BatchedLongShortEnv.

        Args:
            actions (torch.Tensor): The actions value.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The computed or requested result.
        """
        actions = torch.as_tensor(actions, dtype=torch.long).to(self.C.device).reshape(-1)
        if actions.numel() != self.B:
            raise ValueError(f"actions must have {self.B} elements, got {actions.numel()}")
        if bool(((actions < 0) | (actions >= self.action_dim)).any().item()):
            raise ValueError(f"Invalid discrete action, expected values in [0, {self.action_dim - 1}]")

        buckets = self.size_buckets.to(device=self.C.device)
        base_actions = torch.zeros(self.B, dtype=torch.long, device=self.C.device)
        frac = torch.zeros(self.B, dtype=self.dtype, device=self.C.device)

        block = self.N * self.K
        short_offset = 1 + block
        close_offset = 1 + 2 * block

        is_long = (actions >= 1) & (actions < short_offset)
        is_short = (actions >= short_offset) & (actions < close_offset)
        is_close = (actions >= close_offset) & (actions < self.action_dim)

        long_z = (actions - 1).clamp(min=0)
        long_asset = (long_z // self.K).clamp(0, self.N - 1)
        long_bucket = long_z % self.K

        short_z = (actions - short_offset).clamp(min=0)
        short_asset = (short_z // self.K).clamp(0, self.N - 1)
        short_bucket = short_z % self.K

        close_z = (actions - close_offset).clamp(min=0)
        close_asset = (close_z // self.K).clamp(0, self.N - 1)
        close_bucket = close_z % self.K

        base_actions = torch.where(is_long, 1 + long_asset, base_actions)
        base_actions = torch.where(is_short, 1 + self.N + short_asset, base_actions)
        base_actions = torch.where(is_close, 1 + 2 * self.N + close_asset, base_actions)

        frac = torch.where(is_long, buckets[long_bucket], frac)
        frac = torch.where(is_short, buckets[short_bucket], frac)
        frac = torch.where(is_close, buckets[close_bucket], frac)

        return base_actions, frac

    # ------------------------------------------------------------------
    # trading
    # ------------------------------------------------------------------

    def _trade(self, actions: torch.Tensor) -> None:
        """Trade for BatchedLongShortEnv.

        Args:
            actions (torch.Tensor): The actions value.

        Returns:
            None: This function does not return a value.
        """
        base_actions, frac = self._split_bucket_actions(actions)
        super()._trade((base_actions, frac))

    # ------------------------------------------------------------------
    # mask
    # ------------------------------------------------------------------

    def valid_action_mask(self) -> torch.Tensor:
        """Valid action mask for BatchedLongShortEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        B, N = self.B, self.N
        eps = self.eps

        mask = torch.zeros(B, self.action_dim, dtype=torch.bool, device=self._state_device())
        mask[:, 0] = True

        is_long = self.pos_units > eps
        is_short = self.pos_units < -eps
        is_flat = ~(is_long | is_short)

        notional_full = torch.abs(self.pos_units) * self.p
        close_ok_full = (~is_flat) & (notional_full > self.c_fee)

        pnl = self.pos_units * (self.p - self.entry_price)
        gross = self.committed + pnl
        proceeds = (gross - self.c_fee).clamp(min=0.0)
        pre_tax = proceeds - self.committed
        tax = pre_tax.clamp(min=0.0) * self.tax_rate
        net_proc = proceeds - tax

        cash_if_close = torch.where(
            close_ok_full,
            self.C[:, None] + net_proc,
            self.C[:, None].expand(B, N),
        )
        cash_avail = torch.where(is_flat, self.C[:, None].expand(B, N), cash_if_close)

        for b_idx in range(self.K):
            frac = float(self.size_buckets[b_idx].item())
            budget = frac * cash_avail
            can_open = (budget > self.o_fee) & ((budget - self.o_fee) >= self.min_open_dollars)

            long_valid = (~is_long) & can_open & (is_flat | (is_short & close_ok_full))
            short_valid = (~is_short) & can_open & (is_flat | (is_long & close_ok_full))
            close_valid = (~is_flat) & (frac * notional_full > self.c_fee)

            for k in range(self.N):
                mask[:, self.encode_long(k, b_idx)] = long_valid[:, k]
                mask[:, self.encode_short(k, b_idx)] = short_valid[:, k]
                mask[:, self.encode_close(k, b_idx)] = close_valid[:, k]

        return mask


__all__ = [
    "State",
    "StateHistory",
    "LongShortEnv",
    "BatchedLongShortEnv",
]
