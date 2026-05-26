"""Longshort hierarchical leverage utilities for trading environment state, action, reward, and simulation logic."""
from typing import Dict, Tuple

import torch

from rl_trading_playground.environment.longshort_hierarchical import (
    _split_hier_action,
    _split_hier_actions_batched,
)
from rl_trading_playground.environment.generic.longshort_leverage import (
    BatchedLeveragedMultiCurrencyEnv,
    LeveragedMultiCurrencyEnv,
    State,
    StateHistory,
)


class LongShortHierarchicalLeverageEnv(LeveragedMultiCurrencyEnv):
    """
    Hierarchical bucketized long/short trading environment with fixed leverage.

    Hybrid action: (a_d, a_q)
        a_d in [0, 3N]:
            0                          -> hold
            1   .. N                   -> BUY    asset i = a_d - 1
            N+1 .. 2N                  -> SELL   asset i = a_d - 1 - N
            2N+1 .. 3N                 -> CLOSE  asset i = a_d - 1 - 2N
        a_q in [0, K-1]:
            size bucket index, consumed only when buying or selling.

    The actor network is expected to emit ``1 + 3*N + 2*K`` logits, split as:
        [primary (1+3N) | buy_bucket (K) | sell_bucket (K)]
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
        max_leverage: float = 2.0,
        maintenance_margin_ratio: float | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
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
            max_leverage (float): The max leverage value. Defaults to ``2.0``.
            maintenance_margin_ratio (float | None): The maintenance margin ratio value. Defaults to ``None``.
            dtype (torch.dtype): The dtype value. Defaults to ``torch.float32``.
            device (str | torch.device | None): The device value. Defaults to ``None``.
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
            max_leverage=max_leverage,
            maintenance_margin_ratio=maintenance_margin_ratio,
            dtype=dtype,
            device=device,
            eps=eps,
        )

        if len(size_buckets) == 0:
            raise ValueError("size_buckets must not be empty.")
        if not all(0.0 < x <= 1.0 for x in size_buckets):
            raise ValueError("all size buckets must be in (0, 1].")

        self.size_buckets = torch.tensor(size_buckets, dtype=self.dtype, device=self.tau_p.device)
        self.K = int(self.size_buckets.numel())
        self.primary_action_dim = 1 + 3 * self.N
        self.action_dim = self.primary_action_dim + 2 * self.K

    def valid_action_mask(self) -> Dict[str, torch.Tensor]:
        """Valid action mask for LongShortHierarchicalLeverageEnv.

        Returns:
            Dict[str, torch.Tensor]: The computed or requested result.
        """
        device = self._state_device()
        N, K = self.N, self.K

        primary = torch.zeros(self.primary_action_dim, dtype=torch.bool, device=device)
        buy_mask = torch.zeros(N, K, dtype=torch.bool, device=device)
        sell_mask = torch.zeros(N, K, dtype=torch.bool, device=device)
        primary[0] = True

        cash = float(self.C.item())
        portfolio_value = float(self.V.item())
        gross_exposure = float(self.gross_exposure.item())

        for k in range(N):
            units = float(self.pos_units[k].item())
            is_long = units > self.eps
            is_short = units < -self.eps
            is_flat = not (is_long or is_short)

            close_ok_full, cash_avail, value_avail, gross_avail = self._full_close_snapshot(k)
            if is_flat:
                cash_avail = cash
                value_avail = portfolio_value
                gross_avail = gross_exposure

            any_buy = False
            any_sell = False
            for b_idx in range(K):
                frac = float(self.size_buckets[b_idx].item())
                can_open = self._can_open_from_snapshot(
                    cash_available=cash_avail,
                    portfolio_value=value_avail,
                    gross_exposure=gross_avail,
                    frac=frac,
                )

                if is_long:
                    buy_valid = False
                elif is_short:
                    buy_valid = close_ok_full and can_open
                else:
                    buy_valid = can_open
                buy_mask[k, b_idx] = buy_valid
                any_buy = any_buy or buy_valid

                if is_short:
                    sell_valid = False
                elif is_long:
                    sell_valid = close_ok_full and can_open
                else:
                    sell_valid = can_open
                sell_mask[k, b_idx] = sell_valid
                any_sell = any_sell or sell_valid

            primary[1 + k] = any_buy
            primary[1 + N + k] = any_sell
            primary[1 + 2 * N + k] = close_ok_full

        return {"primary": primary, "buy": buy_mask, "sell": sell_mask}

    def _trade(self, a) -> tuple[torch.Tensor, bool]:
        """Trade for LongShortHierarchicalLeverageEnv.

        Args:
            a (Any): The a value.

        Returns:
            tuple[torch.Tensor, bool]: The computed or requested result.
        """
        self.realized_cost.zero_()
        self.realized_pnl.zero_()

        a_d, a_q = _split_hier_action(a)

        if not (0 <= a_d < self.primary_action_dim):
            raise ValueError(f"Invalid primary action {a_d}, expected in [0, {self.primary_action_dim - 1}].")
        if not (0 <= a_q < self.K):
            raise ValueError(f"Invalid bucket action {a_q}, expected in [0, {self.K - 1}].")

        action_info = torch.tensor([0.0, -1.0, 0.0], dtype=self.dtype, device=self._state_device())

        if a_d == 0:
            return action_info, True

        N = self.N
        if 1 <= a_d <= N:
            kind = "long"
            k = a_d - 1
            frac = float(self.size_buckets[a_q].item())
        elif N + 1 <= a_d <= 2 * N:
            kind = "short"
            k = a_d - 1 - N
            frac = float(self.size_buckets[a_q].item())
        else:
            kind = "close"
            k = a_d - 1 - 2 * N
            frac = 1.0

        action_info[1] = float(k)
        action_info[2] = frac

        if kind == "close":
            action_info[0] = 3.0
            did = self._close_position(k, frac)
            self._cleanup_dust()
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
        self._cleanup_dust()
        return action_info, did


class BatchedLongShortHierarchicalLeverageEnv(BatchedLeveragedMultiCurrencyEnv):
    """
    Fully batched hierarchical leveraged long/short environment.

    Action: tuple ``(a_d, a_q)`` of two ``(B,)`` long tensors, or a single
    ``(B, 2)`` long tensor with ``[..., 0] = a_d`` and ``[..., 1] = a_q``.
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
        max_leverage: float = 2.0,
        maintenance_margin_ratio: float | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
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
            max_leverage (float): The max leverage value. Defaults to ``2.0``.
            maintenance_margin_ratio (float | None): The maintenance margin ratio value. Defaults to ``None``.
            dtype (torch.dtype): The dtype value. Defaults to ``torch.float32``.
            device (str | torch.device | None): The device value. Defaults to ``None``.
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
            max_leverage=max_leverage,
            maintenance_margin_ratio=maintenance_margin_ratio,
            dtype=dtype,
            device=device,
            eps=eps,
        )

        if len(size_buckets) == 0:
            raise ValueError("size_buckets must not be empty.")
        if not all(0.0 < x <= 1.0 for x in size_buckets):
            raise ValueError("all size buckets must be in (0, 1].")

        self.size_buckets = torch.tensor(size_buckets, dtype=self.dtype, device=self.tau_p.device)
        self.K = int(self.size_buckets.numel())
        self.primary_action_dim = 1 + 3 * self.N
        self.action_dim = self.primary_action_dim + 2 * self.K

    def _trade(self, actions) -> None:
        """Trade for BatchedLongShortHierarchicalLeverageEnv.

        Args:
            actions (Any): The actions value.

        Returns:
            None: This function does not return a value.
        """
        device = self._state_device()
        a_d, a_q = _split_hier_actions_batched(actions, self.B, device, self.dtype)

        if bool(((a_d < 0) | (a_d >= self.primary_action_dim)).any().item()):
            raise ValueError(f"Invalid primary actions, expected in [0, {self.primary_action_dim - 1}].")
        if bool(((a_q < 0) | (a_q >= self.K)).any().item()):
            raise ValueError(f"Invalid bucket actions, expected in [0, {self.K - 1}].")

        N = self.N
        sell_offset = 1 + N
        close_offset = 1 + 2 * N

        is_buy_act = (a_d >= 1) & (a_d < sell_offset)
        is_sell_act = (a_d >= sell_offset) & (a_d < close_offset)
        is_close_act = (a_d >= close_offset) & (a_d < self.primary_action_dim)

        buckets = self.size_buckets.to(device=device)
        bucket_frac = buckets[a_q]

        base_actions = torch.zeros(self.B, dtype=torch.long, device=device)
        base_actions = torch.where(is_buy_act, 1 + (a_d - 1).clamp(0, N - 1), base_actions)
        base_actions = torch.where(is_sell_act, 1 + N + (a_d - sell_offset).clamp(0, N - 1), base_actions)
        base_actions = torch.where(is_close_act, 1 + 2 * N + (a_d - close_offset).clamp(0, N - 1), base_actions)

        frac = torch.where(
            is_close_act,
            torch.ones(self.B, dtype=self.dtype, device=device),
            torch.where(is_buy_act | is_sell_act, bucket_frac, torch.zeros(self.B, dtype=self.dtype, device=device)),
        )

        super()._trade((base_actions, frac))

    def valid_action_mask(self) -> Dict[str, torch.Tensor]:
        """Valid action mask for BatchedLongShortHierarchicalLeverageEnv.

        Returns:
            Dict[str, torch.Tensor]: The computed or requested result.
        """
        B, N, K = self.B, self.N, self.K
        device = self._state_device()

        snapshots = self._full_close_snapshots()
        buckets = self.size_buckets.to(device=device)
        frac = buckets[None, None, :]

        can_open = self._open_is_valid(
            cash_available=snapshots["cash_after_close"][:, :, None],
            portfolio_value=snapshots["value_after_close"][:, :, None],
            gross_exposure=snapshots["gross_after_close"][:, :, None],
            frac=frac,
        )

        buy_mask = (~snapshots["is_long"][:, :, None]) & can_open & (
            snapshots["is_flat"][:, :, None] | (snapshots["is_short"][:, :, None] & snapshots["close_ok_full"][:, :, None])
        )
        sell_mask = (~snapshots["is_short"][:, :, None]) & can_open & (
            snapshots["is_flat"][:, :, None] | (snapshots["is_long"][:, :, None] & snapshots["close_ok_full"][:, :, None])
        )

        primary = torch.zeros(B, self.primary_action_dim, dtype=torch.bool, device=device)
        primary[:, 0] = True
        primary[:, 1:1 + N] = buy_mask.any(dim=-1)
        primary[:, 1 + N:1 + 2 * N] = sell_mask.any(dim=-1)
        primary[:, 1 + 2 * N:1 + 3 * N] = snapshots["close_ok_full"]
        return {"primary": primary, "buy": buy_mask, "sell": sell_mask}


__all__ = [
    "State",
    "StateHistory",
    "LongShortHierarchicalLeverageEnv",
    "BatchedLongShortHierarchicalLeverageEnv",
]
