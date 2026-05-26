r"""Hybrid-action long/short multi-currency environment with fixed leverage.

Extends [rl_trading_playground.environment.generic.longshort](longshort.html) by introducing a
fixed maximum leverage $\ell_{\max}$ and a maintenance-margin
liquidation rule. Posted collateral $I_k$ is now interpreted as
*margin*, not gross exposure, so gross notional per asset can scale up
to $\ell_{\max} I_k$.

State space
===========

The flat state vector has dimension

$$
\dim(\mathcal{S}) = 4 M N + 9 N + 10,
$$

with the same per-asset block as the unleveraged env and two extra
global features for leverage diagnostics:

$$
\text{globals}_t =
\big[\, t_{\text{vec}},\; c_{\text{rel}},\; \rho_t,\;
\ell^{\mathrm g}_t,\; b_t \,\big],
\qquad
\ell^{\mathrm g}_t, b_t \in \mathbb{R}.
$$

Gross leverage is

$$
\ell^{\mathrm g}_t =
\frac{\sum_k |u_k| p_k}{V_t},
\qquad
V_t = C_t + \sum_k u_k p_k - \text{(unrealised loss reserve)},
$$

and the margin buffer

$$
b_t = \frac{V_t - \mu \sum_k |u_k| p_k}{V_t}
$$

measures distance to liquidation under maintenance-margin ratio
$\mu \in (0, 1)$. By default $\mu = 0.5 / \ell_{\max}$. The agent is
liquidated when $b_t \le 0$.

See [rl_trading_playground.environment.generic.longshort](longshort.html) for the per-asset
block and candle feature definitions; they are inherited unchanged.

Action space
============

Identical to the unleveraged env (hybrid discrete + continuous):

$$
a^{\mathrm d} \in \{0, 1, \dots, 3N\},\qquad a^{\mathrm c} \in [0, 1].
$$

Opening commits $a^{\mathrm c} \cdot C_t$ of cash as *margin*, taking on
gross notional $\ell_{\max} \cdot a^{\mathrm c} \cdot C_t$ on the target
asset. Closing reduces the position by $a^{\mathrm c} \cdot |u_k|$
units.

Reward and termination
======================

Reward modes match the unleveraged env. The episode terminates either
on bankruptcy ($V_t \le V_{\text{bk}}$) or on maintenance-margin
breach ($b_t \le 0$), in both cases applying the penalty
$-\lambda_{\text{done}}$.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from rl_trading_playground.environment.generic.longshort import (
    BatchedMultiCurrencyEnv,
    MultiCurrencyEnv,
)


@dataclass
class State:
    """
    Leveraged hybrid long/short trading state.

    Flattened layout of `to_tensor()`:
        [ globals(10) | asset_0(4M+9) | asset_1(4M+9) | ... | asset_{N-1}(4M+9) ]

    Globals (10):
        time(6), cash_rel(1), rho(1), gross_leverage(1), margin_buffer(1)
    """

    # globals
    time: torch.Tensor
    cash_rel: torch.Tensor
    rho: torch.Tensor
    gross_leverage: torch.Tensor
    margin_buffer: torch.Tensor

    # per-asset
    p_rel: torch.Tensor
    v_rel: torch.Tensor
    vol_rel: torch.Tensor
    body_smooth: torch.Tensor
    hl_rel: torch.Tensor
    body_rel: torch.Tensor
    ret_rel: torch.Tensor
    upper_wick_rel: torch.Tensor
    lower_wick_rel: torch.Tensor
    x_rel: torch.Tensor
    c_rel: torch.Tensor
    unrl_rel: torch.Tensor
    side_rel: torch.Tensor

    def to_tensor(self) -> torch.Tensor:
        """Convert the value to tensor.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        globals_block = torch.cat([
            self.time.flatten(),
            self.cash_rel.reshape(1),
            self.rho.reshape(1),
            self.gross_leverage.reshape(1),
            self.margin_buffer.reshape(1),
        ])
        mn = torch.cat([self.p_rel, self.v_rel, self.vol_rel, self.body_smooth], dim=0).transpose(0, 1)
        scalars = torch.stack([
            self.hl_rel,
            self.body_rel,
            self.ret_rel,
            self.upper_wick_rel,
            self.lower_wick_rel,
            self.x_rel,
            self.c_rel,
            self.unrl_rel,
            self.side_rel,
        ], dim=1)
        asset_block = torch.cat([mn, scalars], dim=1)
        return torch.cat([globals_block, asset_block.flatten()])


class StateHistory:
    """StateHistory implementation for trading environment state, action, reward, and simulation logic."""
    def __init__(self) -> None:
        """Initialize the instance.

        Returns:
            None: This function does not return a value.
        """
        self.states: list[State] = []

    def append(self, state: State) -> None:
        """Append for StateHistory.

        Args:
            state (State): The state value.

        Returns:
            None: This function does not return a value.
        """
        self.states.append(state)

    def get_time(self) -> torch.Tensor:
        """Return the time.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.time for s in self.states])

    def get_cash_rel(self) -> torch.Tensor:
        """Return the cash rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.cash_rel for s in self.states])

    def get_rho(self) -> torch.Tensor:
        """Return the rho.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.rho for s in self.states])

    def get_gross_leverage(self) -> torch.Tensor:
        """Return the gross leverage.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.gross_leverage for s in self.states])

    def get_margin_buffer(self) -> torch.Tensor:
        """Return the margin buffer.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.margin_buffer for s in self.states])

    def get_p_rel(self) -> torch.Tensor:
        """Return the p rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.p_rel for s in self.states])

    def get_v_rel(self) -> torch.Tensor:
        """Return the v rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.v_rel for s in self.states])

    def get_vol_rel(self) -> torch.Tensor:
        """Return the vol rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.vol_rel for s in self.states])

    def get_body_smooth(self) -> torch.Tensor:
        """Return the body smooth.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.body_smooth for s in self.states])

    def get_hl_rel(self) -> torch.Tensor:
        """Return the hl rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.hl_rel for s in self.states])

    def get_body_rel(self) -> torch.Tensor:
        """Return the body rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.body_rel for s in self.states])

    def get_ret_rel(self) -> torch.Tensor:
        """Return the ret rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.ret_rel for s in self.states])

    def get_upper_wick_rel(self) -> torch.Tensor:
        """Return the upper wick rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.upper_wick_rel for s in self.states])

    def get_lower_wick_rel(self) -> torch.Tensor:
        """Return the lower wick rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.lower_wick_rel for s in self.states])

    def get_x_rel(self) -> torch.Tensor:
        """Return the x rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.x_rel for s in self.states])

    def get_c_rel(self) -> torch.Tensor:
        """Return the c rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.c_rel for s in self.states])

    def get_unrl_rel(self) -> torch.Tensor:
        """Return the unrl rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.unrl_rel for s in self.states])

    def get_side_rel(self) -> torch.Tensor:
        """Return the side rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.side_rel for s in self.states])

    def to_tensor(self) -> torch.Tensor:
        """Convert the value to tensor.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.to_tensor() for s in self.states])


class LeveragedMultiCurrencyEnv(MultiCurrencyEnv):
    """
    Long/short environment with fixed leverage and maintenance-margin liquidation.

    `committed` tracks posted margin, not gross exposure.
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
            dtype=dtype,
            device=device,
            eps=eps,
        )
        if max_leverage <= 0.0:
            raise ValueError(f"max_leverage must be > 0, got {max_leverage}.")
        if maintenance_margin_ratio is None:
            maintenance_margin_ratio = 0.5 / float(max_leverage)
        if maintenance_margin_ratio <= 0.0:
            raise ValueError(
                f"maintenance_margin_ratio must be > 0, got {maintenance_margin_ratio}."
            )

        self.max_leverage = float(max_leverage)
        self.maintenance_margin_ratio = float(maintenance_margin_ratio)
        self.state_dim = 4 * self.M * self.N + 9 * self.N + 10
        if self.save_history:
            self.history = StateHistory()

    @property
    def gross_exposure(self) -> torch.Tensor:
        """Gross exposure for LeveragedMultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return (torch.abs(self.pos_units) * self.p).sum()

    @property
    def gross_leverage(self) -> torch.Tensor:
        """Gross leverage for LeveragedMultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return self.gross_exposure / self.V.clamp(min=self.eps)

    @property
    def maintenance_requirement(self) -> torch.Tensor:
        """Maintenance requirement for LeveragedMultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return self.maintenance_margin_ratio * self.gross_exposure

    @property
    def margin_buffer(self) -> torch.Tensor:
        """Margin buffer for LeveragedMultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        V = self.V
        return (V - self.maintenance_requirement) / V.clamp(min=self.eps)

    def _get_state(self) -> State:
        """Return the current state snapshot.

        Returns:
            State: The computed or requested result.
        """
        t_vec = self._compute_time_vector().to(self.dtype)
        cash_rel, x_rel = self._compute_value_weights()
        c_rel, rho = self._compute_cost_weights()
        unrl_rel = self._compute_unrl_rel()
        side_rel = self._compute_side_rel()
        return State(
            time=t_vec,
            cash_rel=cash_rel.to(self.dtype),
            rho=rho.to(self.dtype),
            gross_leverage=self.gross_leverage.to(self.dtype),
            margin_buffer=self.margin_buffer.to(self.dtype),
            p_rel=self.p_rel.clone(),
            v_rel=self.v_rel.clone(),
            vol_rel=self.vol_rel.clone(),
            body_smooth=self.body_smooth.clone(),
            hl_rel=self.hl_rel.clone(),
            body_rel=self.body_rel.clone(),
            ret_rel=self.ret_rel.clone(),
            upper_wick_rel=self.upper_wick_rel.clone(),
            lower_wick_rel=self.lower_wick_rel.clone(),
            x_rel=x_rel.to(self.dtype),
            c_rel=c_rel,
            unrl_rel=unrl_rel,
            side_rel=side_rel,
        )

    def reset(
        self,
        data_or_close=None,
        high=None,
        low=None,
        volume=None,
        time=None,
        C0: Optional[float] | None = None,
        open_=None,
        *,
        data: dict | None = None,
    ) -> State:
        """Reset internal state for a new episode or stream.

        Args:
            data_or_close (Any): The data or close value. Defaults to ``None``.
            high (Any): The high value. Defaults to ``None``.
            low (Any): The low value. Defaults to ``None``.
            volume (Any): The volume value. Defaults to ``None``.
            time (Any): The time value. Defaults to ``None``.
            C0 (Optional[float]): The c0 value. Defaults to ``None``.
            open_ (Any): The open value. Defaults to ``None``.
            data (dict | None): The data value. Defaults to ``None``.

        Returns:
            State: The computed or requested result.
        """
        state = super().reset(
            data_or_close=data_or_close,
            high=high,
            low=low,
            volume=volume,
            time=time,
            C0=C0,
            open_=open_,
            data=data,
        )
        if self.save_history:
            self.history = StateHistory()
            self.history.append(state)
        return state

    def _can_open_from_snapshot(
        self,
        cash_available: float,
        portfolio_value: float,
        gross_exposure: float,
        frac: float,
    ) -> bool:
        """Can open from snapshot for LeveragedMultiCurrencyEnv.

        Args:
            cash_available (float): The cash available value.
            portfolio_value (float): The portfolio value value.
            gross_exposure (float): The gross exposure value.
            frac (float): The frac value.

        Returns:
            bool: The computed or requested result.
        """
        budget = frac * cash_available
        margin = budget - self.o_fee
        if not (budget > self.o_fee and margin >= self.min_open_dollars):
            return False

        value_after_open = portfolio_value - self.o_fee
        gross_after_open = gross_exposure + margin * self.max_leverage
        maintenance_after_open = self.maintenance_margin_ratio * gross_after_open
        return value_after_open + self.eps >= maintenance_after_open

    def _full_close_snapshot(self, k: int) -> tuple[bool, float, float, float]:
        """Full close snapshot for LeveragedMultiCurrencyEnv.

        Args:
            k (int): The k value.

        Returns:
            tuple[bool, float, float, float]: The computed or requested result.
        """
        cash = float(self.C.item())
        portfolio_value = float(self.V.item())
        gross_exposure = float(self.gross_exposure.item())

        units = float(self.pos_units[k].item())
        price = float(self.p[k].item())
        notional_full = abs(units) * price
        close_ok = abs(units) > self.eps and notional_full > self.c_fee
        if not close_ok:
            return False, cash, portfolio_value, gross_exposure

        committed_k = float(self.committed[k].item())
        entry = float(self.entry_price[k].item())
        pnl = units * (price - entry)
        gross = committed_k + pnl
        proceeds = max(gross - self.c_fee, 0.0)
        pre_tax = proceeds - committed_k
        tax = max(pre_tax, 0.0) * self.tax_rate
        net_proceeds = proceeds - tax

        cash_after_close = cash + net_proceeds
        value_after_close = portfolio_value + net_proceeds - committed_k - pnl
        gross_after_close = max(gross_exposure - notional_full, 0.0)
        return True, cash_after_close, value_after_close, gross_after_close

    def _cleanup_dust(self) -> None:
        """Cleanup dust for LeveragedMultiCurrencyEnv.

        Returns:
            None: This function does not return a value.
        """
        dust = self.committed < self.transaction_eps
        self.pos_units[dust] = 0.0
        self.committed[dust] = 0.0
        self.entry_price[dust] = 0.0
        self.C = torch.clamp(self.C, min=0.0)

    def _close_position(self, k: int, frac: float = 1.0, *, force: bool = False) -> bool:
        """Close position for LeveragedMultiCurrencyEnv.

        Args:
            k (int): The k value.
            frac (float): The frac value. Defaults to ``1.0``.
            force (bool): The force value. Defaults to ``False``.

        Returns:
            bool: The computed or requested result.
        """
        units = self.pos_units[k]
        if abs(float(units.item())) <= self.eps:
            return False

        frac = min(max(float(frac), 0.0), 1.0)
        units_to_close = frac * units
        price = self.p[k]
        notional = torch.abs(units_to_close) * price
        if (not force) and float(notional.item()) <= self.c_fee:
            return False

        pos_frac = (torch.abs(units_to_close) / torch.abs(units).clamp(min=self.eps)).clamp(0.0, 1.0)
        realized_committed = pos_frac * self.committed[k]

        pnl = units_to_close * (price - self.entry_price[k])
        gross = realized_committed + pnl
        proceeds = (gross - self.c_fee).clamp(min=0.0)
        pre_tax_pnl = proceeds - realized_committed
        tax = pre_tax_pnl.clamp(min=0.0) * self.tax_rate
        net_proceeds = proceeds - tax
        realized_pnl = net_proceeds - realized_committed

        self.C += net_proceeds
        self.pos_units[k] -= units_to_close
        self.committed[k] -= realized_committed
        self.realized_cost[k] += realized_committed
        self.realized_pnl[k] += realized_pnl
        return True

    def _open_position(self, k: int, side: int, frac: float = 1.0) -> bool:
        """Open position for LeveragedMultiCurrencyEnv.

        Args:
            k (int): The k value.
            side (int): The side value.
            frac (float): The frac value. Defaults to ``1.0``.

        Returns:
            bool: The computed or requested result.
        """
        cash = float(self.C.item())
        if not self._can_open_from_snapshot(
            cash_available=cash,
            portfolio_value=float(self.V.item()),
            gross_exposure=float(self.gross_exposure.item()),
            frac=frac,
        ):
            return False

        budget = frac * cash
        margin = budget - self.o_fee

        price_k = self.p[k].clamp(min=self.eps)
        margin_t = torch.tensor(margin, dtype=self.dtype, device=self.C.device)
        open_fee_t = torch.tensor(self.o_fee, dtype=self.dtype, device=self.C.device)
        leverage_t = torch.tensor(self.max_leverage, dtype=self.dtype, device=self.C.device)

        notional_t = margin_t * leverage_t
        units_mag = notional_t / price_k
        total_cost = torch.minimum(margin_t + open_fee_t, self.C)

        self.C -= total_cost
        self.pos_units[k] += float(side) * units_mag
        self.committed[k] += margin_t

        old_units = abs(float(self.pos_units[k].item())) - float(units_mag.item())
        if old_units > self.eps:
            old_entry = float(self.entry_price[k].item())
            total_units = abs(float(self.pos_units[k].item()))
            new_entry = (old_units * old_entry + float(units_mag.item()) * float(price_k.item())) / total_units
            self.entry_price[k] = torch.tensor(new_entry, dtype=self.dtype, device=self.C.device)
        else:
            self.entry_price[k] = self.p[k].clone()
        return True

    def _liquidate_all_positions(self) -> bool:
        """Liquidate all positions for LeveragedMultiCurrencyEnv.

        Returns:
            bool: The computed or requested result.
        """
        liquidated = False
        for k in range(self.N):
            liquidated = self._close_position(k, frac=1.0, force=True) or liquidated
        self._cleanup_dust()
        return liquidated

    def valid_action_mask(self) -> torch.Tensor:
        """Valid action mask for LeveragedMultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self._state_device())
        mask[0] = True

        cash = float(self.C.item())
        portfolio_value = float(self.V.item())
        gross_exposure = float(self.gross_exposure.item())

        for k in range(self.N):
            units = float(self.pos_units[k].item())
            is_long = units > self.eps
            is_short = units < -self.eps
            is_flat = not (is_long or is_short)

            close_ok, cash_avail, value_avail, gross_avail = self._full_close_snapshot(k)
            if is_flat:
                cash_avail = cash
                value_avail = portfolio_value
                gross_avail = gross_exposure

            can_open = self._can_open_from_snapshot(cash_avail, value_avail, gross_avail, frac=1.0)

            if is_long:
                mask[self.encode_long(k)] = False
            elif is_short:
                mask[self.encode_long(k)] = close_ok and can_open
            else:
                mask[self.encode_long(k)] = can_open

            if is_short:
                mask[self.encode_short(k)] = False
            elif is_long:
                mask[self.encode_short(k)] = close_ok and can_open
            else:
                mask[self.encode_short(k)] = can_open

            mask[self.encode_close(k)] = close_ok

        return mask

    def step(
        self,
        a,
        data_or_close=None,
        high=None,
        low=None,
        volume=None,
        time=None,
        open_=None,
        *,
        data: dict | None = None,
    ) -> Tuple[State, float, bool, Dict]:
        """Advance the simulation by one step.

        Args:
            a (Any): The a value.
            data_or_close (Any): The data or close value. Defaults to ``None``.
            high (Any): The high value. Defaults to ``None``.
            low (Any): The low value. Defaults to ``None``.
            volume (Any): The volume value. Defaults to ``None``.
            time (Any): The time value. Defaults to ``None``.
            open_ (Any): The open value. Defaults to ``None``.
            data (dict | None): The data value. Defaults to ``None``.

        Returns:
            Tuple[State, float, bool, Dict]: The computed or requested result.
        """
        action_info, valid_trade = self._trade(a)
        self._update(
            close_or_data=data_or_close,
            high=high,
            low=low,
            volume=volume,
            time=time,
            open_=open_,
            data=data,
        )

        margin_breach = bool((self.V < self.maintenance_requirement).item())
        liquidated = self._liquidate_all_positions() if margin_breach else False
        reward = self._reward()

        done = float(self.V.item()) <= self.bankruptcy_threshold
        if done:
            reward -= self.done_reward_penalty

        info = {
            "t": self.t,
            "action_type": int(action_info[0].item()),
            "action_asset": int(action_info[1].item()),
            "action_frac": float(action_info[2].item()),
            "valid_trade": bool(valid_trade),
            "liquidated": bool(liquidated),
            "margin_breach": bool(margin_breach),
            "C": float(self.C.item()),
            "V": float(self.V.item()),
            "gross_exposure": float(self.gross_exposure.item()),
            "gross_leverage": float(self.gross_leverage.item()),
            "maintenance_requirement": float(self.maintenance_requirement.item()),
            "margin_buffer": float(self.margin_buffer.item()),
            "pos_units": self.pos_units.tolist(),
            "committed": self.committed.tolist(),
            "entry_price": self.entry_price.tolist(),
            "p": self.p.tolist(),
            "realized_cost": self.realized_cost.tolist(),
            "realized_pnl": self.realized_pnl.tolist(),
        }

        next_state = self._get_state()
        if self.save_history and self.history is not None:
            self.history.append(next_state)
        return next_state, reward, done, info


class BatchedLeveragedMultiCurrencyEnv(BatchedMultiCurrencyEnv):
    """BatchedLeveragedMultiCurrencyEnv implementation for trading environment state, action, reward, and simulation logic."""
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
            dtype=dtype,
            device=device,
            eps=eps,
        )
        if max_leverage <= 0.0:
            raise ValueError(f"max_leverage must be > 0, got {max_leverage}.")
        if maintenance_margin_ratio is None:
            maintenance_margin_ratio = 0.5 / float(max_leverage)
        if maintenance_margin_ratio <= 0.0:
            raise ValueError(
                f"maintenance_margin_ratio must be > 0, got {maintenance_margin_ratio}."
            )

        self.max_leverage = float(max_leverage)
        self.maintenance_margin_ratio = float(maintenance_margin_ratio)
        self.state_dim = 4 * self.M * self.N + 9 * self.N + 10

    @property
    def gross_exposure(self) -> torch.Tensor:
        """Gross exposure for BatchedLeveragedMultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return (torch.abs(self.pos_units) * self.p).sum(dim=-1)

    @property
    def gross_leverage(self) -> torch.Tensor:
        """Gross leverage for BatchedLeveragedMultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return self.gross_exposure / self.V.clamp(min=self.eps)

    @property
    def maintenance_requirement(self) -> torch.Tensor:
        """Maintenance requirement for BatchedLeveragedMultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return self.maintenance_margin_ratio * self.gross_exposure

    @property
    def margin_buffer(self) -> torch.Tensor:
        """Margin buffer for BatchedLeveragedMultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        V = self.V
        return (V - self.maintenance_requirement) / V.clamp(min=self.eps)

    def _obs(self):
        """Obs for BatchedLeveragedMultiCurrencyEnv.

        Returns:
            Any: The computed or requested result.
        """
        B, N = self.B, self.N
        V = self.V.clamp(min=self.eps)

        t_vec = self._time_features()
        cash_rel = self.C / V
        x_assets = (self.pos_units * self.p) / V[:, None]

        S = self.committed.sum(dim=1)
        c_rel = torch.where(
            S[:, None] > 0,
            self.committed / (S[:, None] + self.eps),
            torch.zeros(B, N, dtype=self.dtype, device=self._state_device()),
        )
        rho = S / (S + self.C + self.eps)

        has = self.committed > self.eps
        pnl = self.pos_units * (self.p - self.entry_price)
        tax = pnl.clamp(min=0.0) * self.tax_rate
        net = pnl - tax
        unrl_rel = torch.where(
            has,
            net / self.committed.clamp(min=self.eps),
            torch.zeros(B, N, dtype=self.dtype, device=self._state_device()),
        )

        side_rel = torch.zeros(B, N, dtype=self.dtype, device=self._state_device())
        side_rel[self.pos_units > self.eps] = 1.0
        side_rel[self.pos_units < -self.eps] = -1.0

        mn = torch.cat([self.p_rel, self.v_rel, self.vol_rel, self.body_smooth], dim=1).transpose(1, 2)
        scalars = torch.stack([
            self.hl_rel,
            self.body_rel,
            self.ret_rel,
            self.upper_wick_rel,
            self.lower_wick_rel,
            x_assets,
            c_rel,
            unrl_rel,
            side_rel,
        ], dim=2)
        asset_block = torch.cat([mn, scalars], dim=2)

        globals_block = torch.cat([
            t_vec,
            cash_rel[:, None],
            rho[:, None],
            self.gross_leverage[:, None],
            self.margin_buffer[:, None],
        ], dim=1)
        return torch.cat([globals_block, asset_block.reshape(B, -1)], dim=1)

    def _open_is_valid(
        self,
        cash_available: torch.Tensor,
        portfolio_value: torch.Tensor,
        gross_exposure: torch.Tensor,
        frac: torch.Tensor,
    ) -> torch.Tensor:
        """Open is valid for BatchedLeveragedMultiCurrencyEnv.

        Args:
            cash_available (torch.Tensor): The cash available value.
            portfolio_value (torch.Tensor): The portfolio value value.
            gross_exposure (torch.Tensor): The gross exposure value.
            frac (torch.Tensor): The frac value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        budget = frac * cash_available
        margin = budget - self.o_fee
        value_after_open = portfolio_value - self.o_fee
        gross_after_open = gross_exposure + margin * self.max_leverage
        maintenance_after_open = self.maintenance_margin_ratio * gross_after_open
        return (
            (budget > self.o_fee)
            & (margin >= self.min_open_dollars)
            & (value_after_open + self.eps >= maintenance_after_open)
        )

    def _full_close_snapshots(self) -> Dict[str, torch.Tensor]:
        """Full close snapshots for BatchedLeveragedMultiCurrencyEnv.

        Returns:
            Dict[str, torch.Tensor]: The computed or requested result.
        """
        B, N = self.B, self.N
        eps = self.eps

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

        cash_after_close = torch.where(
            close_ok_full,
            self.C[:, None] + net_proc,
            self.C[:, None].expand(B, N),
        )
        value_after_close = torch.where(
            close_ok_full,
            self.V[:, None] + net_proc - self.committed - pnl,
            self.V[:, None].expand(B, N),
        )
        gross_after_close = torch.where(
            close_ok_full,
            (self.gross_exposure[:, None] - notional_full).clamp(min=0.0),
            self.gross_exposure[:, None].expand(B, N),
        )
        return {
            "is_long": is_long,
            "is_short": is_short,
            "is_flat": is_flat,
            "close_ok_full": close_ok_full,
            "cash_after_close": cash_after_close,
            "value_after_close": value_after_close,
            "gross_after_close": gross_after_close,
        }

    def _cleanup_dust(self) -> None:
        """Cleanup dust for BatchedLeveragedMultiCurrencyEnv.

        Returns:
            None: This function does not return a value.
        """
        dust = self.committed < self.transaction_eps
        self.pos_units = self.pos_units.masked_fill(dust, 0.0)
        self.committed = self.committed.masked_fill(dust, 0.0)
        self.entry_price = self.entry_price.masked_fill(dust, 0.0)
        self.C = self.C.clamp(min=0.0)

    def _trade(self, actions) -> None:
        """Trade for BatchedLeveragedMultiCurrencyEnv.

        Args:
            actions (Any): The actions value.

        Returns:
            None: This function does not return a value.
        """
        B, N = self.B, self.N
        b = self._b_idx
        actions_d, frac = self._split_actions(actions)

        self.realized_cost = torch.zeros(B, N, dtype=self.dtype, device=self._state_device())
        self.realized_pnl = torch.zeros(B, N, dtype=self.dtype, device=self._state_device())

        sell_offset = 1 + N
        close_offset = 1 + 2 * N

        is_long_act = (actions_d >= 1) & (actions_d < sell_offset)
        is_short_act = (actions_d >= sell_offset) & (actions_d < close_offset)
        is_close_act = (actions_d >= close_offset) & (actions_d < self.action_dim)

        long_asset = (actions_d - 1).clamp(0, N - 1)
        short_asset = (actions_d - sell_offset).clamp(0, N - 1)
        close_asset = (actions_d - close_offset).clamp(0, N - 1)

        cur_short_on_long = self.pos_units[b, long_asset] < -self.eps
        cur_long_on_short = self.pos_units[b, short_asset] > self.eps

        flip_long = is_long_act & cur_short_on_long
        flip_short = is_short_act & cur_long_on_short
        do_close = is_close_act | flip_long | flip_short

        target_close_asset = torch.where(
            is_close_act,
            close_asset,
            torch.where(flip_long, long_asset, short_asset),
        )

        close_frac = torch.where(
            is_close_act,
            frac,
            torch.ones(B, dtype=self.dtype, device=self._state_device()),
        )

        units_c = self.pos_units[b, target_close_asset]
        entry_c = self.entry_price[b, target_close_asset]
        committed_c = self.committed[b, target_close_asset]
        price_c = self.p[b, target_close_asset]

        units_to_close = close_frac * units_c
        notional_c = torch.abs(units_to_close) * price_c
        pos_frac_c = (torch.abs(units_to_close) / torch.abs(units_c).clamp(min=self.eps)).clamp(0.0, 1.0)
        realized_committed = pos_frac_c * committed_c

        close_valid = do_close & (torch.abs(units_c) > self.eps) & (notional_c > self.c_fee)

        pnl_c = units_to_close * (price_c - entry_c)
        gross_c = realized_committed + pnl_c
        proceeds = (gross_c - self.c_fee).clamp(min=0.0)
        pre_tax_pnl = proceeds - realized_committed
        tax = pre_tax_pnl.clamp(min=0.0) * self.tax_rate
        net_proceeds = proceeds - tax
        realized_pnl = net_proceeds - realized_committed

        cv = close_valid.to(self.dtype)
        self.C = self.C + net_proceeds * cv
        self.pos_units.scatter_add_(1, target_close_asset[:, None], -(units_to_close * cv)[:, None])
        self.committed.scatter_add_(1, target_close_asset[:, None], -(realized_committed * cv)[:, None])
        self.realized_cost.scatter_add_(1, target_close_asset[:, None], (realized_committed * cv)[:, None])
        self.realized_pnl.scatter_add_(1, target_close_asset[:, None], (realized_pnl * cv)[:, None])

        flat_after_close = torch.abs(self.pos_units) <= self.eps
        self.entry_price = self.entry_price.masked_fill(flat_after_close, 0.0)

        flip_failed_long = flip_long & ~close_valid
        flip_failed_short = flip_short & ~close_valid

        is_open = (is_long_act & ~flip_failed_long) | (is_short_act & ~flip_failed_short)
        open_asset = torch.where(is_long_act, long_asset, short_asset)
        open_side = torch.where(
            is_long_act,
            torch.ones(B, dtype=self.dtype, device=self._state_device()),
            -torch.ones(B, dtype=self.dtype, device=self._state_device()),
        )

        cur_units_open = self.pos_units[b, open_asset]
        slot_flat = torch.abs(cur_units_open) <= self.eps

        buy_budget = frac * self.C
        margin = buy_budget - self.o_fee
        open_valid = is_open & slot_flat & self._open_is_valid(
            cash_available=self.C,
            portfolio_value=self.V,
            gross_exposure=self.gross_exposure,
            frac=frac,
        )

        open_price = self.p[b, open_asset].clamp(min=self.eps)
        notional = margin * self.max_leverage
        units_mag = notional / open_price
        open_units = open_side * units_mag
        total_cost = torch.minimum(buy_budget, self.C)

        ov = open_valid.to(self.dtype)
        self.C = self.C - total_cost * ov
        self.pos_units.scatter_add_(1, open_asset[:, None], (open_units * ov)[:, None])
        self.committed.scatter_add_(1, open_asset[:, None], (margin * ov)[:, None])
        self.entry_price.scatter_add_(1, open_asset[:, None], (open_price * ov)[:, None])

        self._cleanup_dust()

    def _liquidate_breached(self, breached: torch.Tensor) -> torch.Tensor:
        """Liquidate breached for BatchedLeveragedMultiCurrencyEnv.

        Args:
            breached (torch.Tensor): The breached value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        to_liquidate = breached[:, None] & (torch.abs(self.pos_units) > self.eps)
        if not bool(to_liquidate.any().item()):
            return torch.zeros(self.B, dtype=torch.bool, device=self._state_device())

        committed = self.committed
        pnl = self.pos_units * (self.p - self.entry_price)
        gross = committed + pnl
        proceeds = (gross - self.c_fee).clamp(min=0.0)
        pre_tax_pnl = proceeds - committed
        tax = pre_tax_pnl.clamp(min=0.0) * self.tax_rate
        net_proceeds = proceeds - tax
        realized_pnl = net_proceeds - committed

        cv = to_liquidate.to(self.dtype)
        self.C = self.C + (net_proceeds * cv).sum(dim=1)
        self.realized_cost = self.realized_cost + committed * cv
        self.realized_pnl = self.realized_pnl + realized_pnl * cv
        self.pos_units = self.pos_units.masked_fill(to_liquidate, 0.0)
        self.committed = self.committed.masked_fill(to_liquidate, 0.0)
        self.entry_price = self.entry_price.masked_fill(to_liquidate, 0.0)

        self._cleanup_dust()
        return to_liquidate.any(dim=1)

    def step(self, actions, open_, close, high, low, volume, time):
        """Advance the simulation by one step.

        Args:
            actions (Any): The actions value.
            open_ (Any): The open value.
            close (Any): The close value.
            high (Any): The high value.
            low (Any): The low value.
            volume (Any): The volume value.
            time (Any): The time value.

        Returns:
            Any: The computed or requested result.
        """
        self._trade(actions)
        self._update(open_, close, high, low, volume, time)
        breached = self.V < self.maintenance_requirement
        if bool(breached.any().item()):
            self._liquidate_breached(breached)
        rewards = self._reward()
        dones = self.V <= self.bankruptcy_threshold
        rewards = torch.where(dones, rewards - self.done_reward_penalty, rewards)
        return self._obs(), rewards, dones

    def valid_action_mask(self):
        """Valid action mask for BatchedLeveragedMultiCurrencyEnv.

        Returns:
            Any: The computed or requested result.
        """
        B, N = self.B, self.N
        snapshots = self._full_close_snapshots()

        mask = torch.zeros(B, self.action_dim, dtype=torch.bool, device=self._state_device())
        mask[:, 0] = True

        can_open = self._open_is_valid(
            cash_available=snapshots["cash_after_close"],
            portfolio_value=snapshots["value_after_close"],
            gross_exposure=snapshots["gross_after_close"],
            frac=torch.ones(B, N, dtype=self.dtype, device=self._state_device()),
        )

        long_valid = (~snapshots["is_long"]) & can_open & (
            snapshots["is_flat"] | (snapshots["is_short"] & snapshots["close_ok_full"])
        )
        short_valid = (~snapshots["is_short"]) & can_open & (
            snapshots["is_flat"] | (snapshots["is_long"] & snapshots["close_ok_full"])
        )

        mask[:, 1:1 + N] = long_valid
        mask[:, 1 + N:1 + 2 * N] = short_valid
        mask[:, 1 + 2 * N:1 + 3 * N] = snapshots["close_ok_full"]
        return mask


__all__ = [
    "State",
    "StateHistory",
    "LeveragedMultiCurrencyEnv",
    "BatchedLeveragedMultiCurrencyEnv",
]
