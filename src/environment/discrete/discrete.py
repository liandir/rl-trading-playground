"""Discrete utilities for trading environment state, action, reward, and simulation logic."""
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch

PI = 3.141592653589793238462


@dataclass
class State:
    """State implementation for trading environment state, action, reward, and simulation logic."""
    time: torch.Tensor            # [6]    cyclic time features
    p_rel: torch.Tensor           # [M, N]
    hl_rel: torch.Tensor          # [N]    symmetric high-low range
    x_rel: torch.Tensor           # [N+1]  exposure: cash + value weights
    c_rel: torch.Tensor           # [N]    invested weights, stable
    rho: torch.Tensor             # []     overall commitment
    v_rel: torch.Tensor           # [M, N]  volume deviation from multi-timescale EMAs
    unrl_rel: torch.Tensor        # [N]     unrealized after-tax PnL / invested (0 when no position)
    vol_rel: torch.Tensor         # [M, N]  hl_rel deviation from multi-timescale EMAs (volatility regime)

    def to_tensor(self):
        """Convert the value to tensor.

        Returns:
            Any: The computed or requested result.
        """
        return torch.cat([
            self.time.flatten(),
            self.p_rel.flatten(),
            self.hl_rel.flatten(),
            self.x_rel.flatten(),
            self.c_rel.flatten(),
            self.rho.view(1),
            self.v_rel.flatten(),
            self.unrl_rel.flatten(),
            self.vol_rel.flatten(),
        ])


class StateHistory:
    """StateHistory implementation for trading environment state, action, reward, and simulation logic."""
    def __init__(self):
        """Initialize the instance.

        Returns:
            None: This function does not return a value.
        """
        self.states: list[State] = []

    def append(self, state: State):
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

    def get_p_rel(self) -> torch.Tensor:
        """Return the p rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.p_rel for s in self.states])

    def get_hl_rel(self) -> torch.Tensor:
        """Return the hl rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.hl_rel for s in self.states])

    def get_x_rel(self) -> torch.Tensor:
        """Return the x rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.x_rel for s in self.states])

    def get_v_rel(self) -> torch.Tensor:
        """Return the v rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.v_rel for s in self.states])

    def get_unrl_rel(self) -> torch.Tensor:
        """Return the unrl rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.unrl_rel for s in self.states])

    def get_vol_rel(self) -> torch.Tensor:
        """Return the vol rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.vol_rel for s in self.states])

    def to_tensor(self) -> torch.Tensor:
        """Convert the value to tensor.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.to_tensor() for s in self.states])


class MultiCurrencyEnv:
    """
    Discrete-action multi-asset trading environment (v00 — no size buckets).

    Action space:
        0       -> hold
        1 .. N  -> toggle asset i:
                     if position open  -> sell all units of asset i
                     if no position    -> spend all cash on asset i

    State features:
        time     [6]      sin/cos encoding of time-of-day, day-of-week, week-of-year
        p_rel    [M, N]   price vs M trend EMAs
        hl_rel   [N]      symmetric high-low range
        x_rel    [N+1]    cash + asset value fractions
        c_rel    [N]      invested weights, stable
        rho      [1]      overall invested fraction
        v_rel    [M, N]   volume deviation from M trend EMAs
        unrl_rel [N]      unrealized after-tax PnL / invested (0 when flat)
        vol_rel  [M, N]   volatility regime across M scales

    state_dim = 3*M*N + 4*N + 8
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
        sell_fee: float = 1.0,
        buy_fee: float = 1.0,
        tax_rate: float = 0.26,
        min_buy_dollars: float = 10.0,
        val_coeff: float = 1.0,
        roi_coeff: float = 1.0,
        reward_mode: str = "log",
        done_reward_penalty: float = 1.0,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ):
        """Initialize the instance.

        Args:
            N (int): The n value.
            C0 (float): The c0 value.
            tau_p (torch.Tensor): The tau p value.
            bankruptcy_threshold (float): The bankruptcy threshold value. Defaults to ``1.0``.
            transaction_eps (float): The transaction eps value. Defaults to ``0.01``.
            use_dollar_volume (bool): The use dollar volume value. Defaults to ``True``.
            save_history (bool): The save history value. Defaults to ``False``.
            sell_fee (float): The sell fee value. Defaults to ``1.0``.
            buy_fee (float): The buy fee value. Defaults to ``1.0``.
            tax_rate (float): The tax rate value. Defaults to ``0.26``.
            min_buy_dollars (float): The min buy dollars value. Defaults to ``10.0``.
            val_coeff (float): The val coeff value. Defaults to ``1.0``.
            roi_coeff (float): The roi coeff value. Defaults to ``1.0``.
            reward_mode (str): The reward mode value. Defaults to ``'log'``.
            done_reward_penalty (float): The done reward penalty value. Defaults to ``1.0``.
            dtype (torch.dtype): The dtype value. Defaults to ``torch.float32``.
            eps (float): The eps value. Defaults to ``1e-08``.

        Returns:
            None: This function does not return a value.
        """
        assert isinstance(N, int) and N > 0
        assert tau_p.ndim == 1 and tau_p.numel() > 0
        assert min_buy_dollars >= buy_fee, "min_buy_dollars must be >= buy_fee"
        assert reward_mode in ("return", "log"), "reward_mode must be one of ('return', 'log')"

        self.N = N
        self.dtype = dtype
        self.eps = float(eps)

        self.bankruptcy_threshold = float(bankruptcy_threshold)
        self.transaction_eps = float(transaction_eps)
        self.use_dollar_volume = bool(use_dollar_volume)
        self.tax_rate = float(tax_rate)
        self.min_buy_dollars = float(min_buy_dollars)
        self.val_coeff = float(val_coeff)
        self.roi_coeff = float(roi_coeff)

        self.reward_mode = reward_mode
        self.done_reward_penalty = float(done_reward_penalty)

        self.s_fee = float(sell_fee)
        self.b_fee = float(buy_fee)

        self.tau_p = tau_p.to(self.dtype)
        self.M = self.tau_p.numel()

        self.save_history = bool(save_history)
        self.history = StateHistory() if self.save_history else None

        self.C0 = float(C0)

        # state = time(6) + p_rel(M*N) + hl_rel(N) + x_rel(N+1) + c_rel(N) + rho(1) + v_rel(M*N) + unrl_rel(N) + vol_rel(M*N)
        self.state_dim = 3 * self.M * N + 4 * N + 8

        # discrete actions: hold + N toggle actions
        self.action_dim = 1 + self.N

        # runtime attributes initialized in reset()
        self.C = None
        self.w = None
        self.t = None
        self.dt = None
        self.p = None
        self.v = None
        self.v_smooth = None
        self.p_smooth = None
        self.p_rel = None
        self.hl_rel = None
        self.hl_smooth = None
        self.v_rel = None
        self.vol_rel = None
        self.V_prev = None
        self.invested = None
        self.realized_cost = None
        self.realized_pnl = None

    # -------------------------------------------------------------------------
    # action helpers
    # -------------------------------------------------------------------------

    def encode_hold(self) -> int:
        """Encode hold for MultiCurrencyEnv.

        Returns:
            int: The computed or requested result.
        """
        return 0

    def encode_asset(self, asset_idx: int) -> int:
        """Encode asset for MultiCurrencyEnv.

        Args:
            asset_idx (int): The asset idx value.

        Returns:
            int: The computed or requested result.
        """
        if not (0 <= asset_idx < self.N):
            raise ValueError("asset_idx out of range")
        return 1 + asset_idx

    def decode_action(self, action_idx: int) -> tuple[str, int | None]:
        """Decode action for MultiCurrencyEnv.

        Args:
            action_idx (int): The action idx value.

        Returns:
            tuple[str, int | None]: The computed or requested result.
        """
        if action_idx == 0:
            return ("hold", None)
        if 1 <= action_idx <= self.N:
            return ("toggle", action_idx - 1)
        raise ValueError(f"invalid action index {action_idx}")

    def valid_action_mask(self) -> torch.Tensor:
        """Boolean mask of shape [action_dim]. True = action is executable.
        Toggle action k is valid if the asset has an open position (sell path)
        or there is enough cash to open one (buy path).

        Returns:
            torch.Tensor: The computed or requested result.
        """
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[0] = True  # hold is always valid

        cash = float(self.C.item())
        can_buy = cash > self.b_fee and (cash - self.b_fee) >= self.min_buy_dollars

        for k in range(self.N):
            units = float(self.w[k].item())
            if units > 0.0:
                sell_value = units * float(self.p[k].item())
                if sell_value > self.s_fee:
                    mask[k + 1] = True
            elif can_buy:
                mask[k + 1] = True

        return mask

    # -------------------------------------------------------------------------
    # state feature computations
    # -------------------------------------------------------------------------

    def _compute_time_vector(self) -> torch.Tensor:
        """Compute the time vector.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        time = datetime.fromtimestamp(self.t)
        dateiso = time.isocalendar()

        total_weeks = datetime(time.year, 12, 28).isocalendar().week

        period_day = (time.hour + (time.minute + time.microsecond / 1e6) / 60.0) / 24.0
        period_week = (dateiso.weekday - 1 + period_day) / 7.0
        period_year = (dateiso.week - 1 + period_week) / max(total_weeks, 1)

        angles = torch.tensor([period_day, period_week, period_year], dtype=self.dtype)
        return torch.cat([torch.sin(2 * PI * angles), torch.cos(2 * PI * angles)])

    def _compute_value_weights(self) -> torch.Tensor:
        """Compute the value weights.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        x_cash = (self.C / self.V)[None]
        x_assets = (self.w * self.p) / self.V
        return torch.cat([x_cash, x_assets])

    def _compute_cost_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the cost weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The computed or requested result.
        """
        S = self.invested.sum()
        c_rel = (self.invested / (S + self.eps)) if S.item() > 0 else torch.zeros(self.N, dtype=self.dtype)
        rho = S / (S + self.C + self.eps)
        return c_rel, rho

    def _compute_unrl_rel(self) -> torch.Tensor:
        """Unrealized after-tax PnL divided by invested cost — zero when no position.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        has = self.invested > self.eps
        unrl_rel = torch.zeros(self.N, dtype=self.dtype)
        if has.any():
            unrl_rel[has] = self._unrealized_after_tax_pnl()[has] / self.invested[has]
        return unrl_rel

    def _compute_hl_rel(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        """Compute the hl rel.

        Args:
            high (torch.Tensor): The high value.
            low (torch.Tensor): The low value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return 2.0 * (high - low) / (high + low + self.eps)

    def _get_market_inputs(self, data: dict) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the market inputs.

        Args:
            data (dict): The data value.

        Returns:
            tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The computed or requested result.
        """
        if "close" not in data:
            raise KeyError("data must contain 'close'")
        if "high" not in data:
            raise KeyError("data must contain 'high'")
        if "low" not in data:
            raise KeyError("data must contain 'low'")
        if "volume" not in data:
            raise KeyError("data must contain 'volume'")

        close  = data["close"].to(self.dtype).reshape(-1)
        high   = data["high"].to(self.dtype).reshape(-1)
        low    = data["low"].to(self.dtype).reshape(-1)
        volume = data["volume"].to(self.dtype).reshape(-1)

        for name, value in (("close", close), ("volume", volume), ("high", high), ("low", low)):
            if value.numel() != self.N:
                raise ValueError(f"data['{name}'] must have {self.N} elements, got {value.numel()}")

        return float(data["time"]), close, high, low, volume

    def _avg_cost_per_unit(self) -> torch.Tensor:
        """Avg cost per unit for MultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        avg = torch.zeros(self.N, dtype=self.dtype)
        has = self.w > self.eps
        avg[has] = self.invested[has] / self.w[has].clamp(min=self.eps)
        return avg

    def _unrealized_after_tax_pnl(self) -> torch.Tensor:
        """Unrealized after tax pnl for MultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        has = self.w > self.eps
        avg_cost = self._avg_cost_per_unit()

        unrl_pre = torch.zeros(self.N, dtype=self.dtype)
        unrl_pre[has] = (self.p[has] - avg_cost[has]) * self.w[has]

        unrl_tax = unrl_pre.clamp(min=0.0) * self.tax_rate
        return unrl_pre - unrl_tax

    def _get_state(self) -> State:
        """Return the current state snapshot.

        Returns:
            State: The computed or requested result.
        """
        t_vec = self._compute_time_vector().to(self.dtype)
        x_rel = self._compute_value_weights().to(self.dtype)
        c_rel, rho = self._compute_cost_weights()
        unrl_rel = self._compute_unrl_rel()

        return State(
            time=t_vec,
            p_rel=self.p_rel.clone(),
            hl_rel=self.hl_rel.clone(),
            v_rel=self.v_rel.clone(),
            x_rel=x_rel,
            c_rel=c_rel,
            rho=rho,
            unrl_rel=unrl_rel,
            vol_rel=self.vol_rel.clone(),
        )

    # -------------------------------------------------------------------------
    # portfolio properties
    # -------------------------------------------------------------------------

    @property
    def V(self) -> torch.Tensor:
        """V for MultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return self.C + torch.sum(self.w * self.p)

    # -------------------------------------------------------------------------
    # reset / update
    # -------------------------------------------------------------------------

    def reset(self, data: dict, C0: Optional[float] = None) -> State:
        """Reset internal state for a new episode or stream.

        Args:
            data (dict): The data value.
            C0 (Optional[float]): The c0 value. Defaults to ``None``.

        Returns:
            State: The computed or requested result.
        """
        if C0 is not None:
            self.C0 = float(C0)

        self.C = torch.tensor(self.C0, dtype=self.dtype)
        self.w = torch.zeros(self.N, dtype=self.dtype)

        t, close, high, low, volume = self._get_market_inputs(data)

        self.t = t
        self.dt = 0.0
        self.p = close.clone()
        self.v = volume.clone()
        if self.use_dollar_volume:
            self.v *= self.p

        self.v_smooth = self.v[None, :].repeat(self.M, 1).clone()

        self.p_smooth = self.p[None, :].repeat(self.M, 1).clone()
        self.p_rel = torch.zeros((self.M, self.N), dtype=self.dtype)
        self.hl_rel = self._compute_hl_rel(high, low)
        self.hl_smooth = self.hl_rel[None, :].repeat(self.M, 1).clone()
        self.v_rel = torch.zeros((self.M, self.N), dtype=self.dtype)
        self.vol_rel = torch.zeros((self.M, self.N), dtype=self.dtype)

        self.V_prev = self.V.detach().clone()

        self.invested = torch.zeros(self.N, dtype=self.dtype)
        self.realized_cost = torch.zeros(self.N, dtype=self.dtype)
        self.realized_pnl = torch.zeros(self.N, dtype=self.dtype)

        state = self._get_state()

        if self.save_history:
            self.history = StateHistory()
            self.history.append(state)

        return state

    def _update(self, data: dict) -> None:
        """Apply one update step.

        Args:
            data (dict): The data value.

        Returns:
            None: This function does not return a value.
        """
        self.V_prev = self.V.detach().clone()

        t_new, close, high, low, volume = self._get_market_inputs(data)
        if t_new < self.t:
            print("WARNING: dt < 0 - setting to zero!")

        self.dt = (t_new - self.t) * float(t_new > self.t)
        self.t = t_new

        self.p[:] = close

        alpha_p = 1 - torch.exp(-self.dt / self.tau_p.clamp(min=self.eps))[:, None]
        self.p_smooth += alpha_p * (self.p[None, :] - self.p_smooth)
        self.p_rel = (self.p[None, :] - self.p_smooth) / self.p_smooth.clamp(min=self.eps)
        self.hl_rel[:] = self._compute_hl_rel(high, low)
        self.hl_smooth += alpha_p * (self.hl_rel[None, :] - self.hl_smooth)
        self.vol_rel = (self.hl_rel[None, :] - self.hl_smooth) / self.hl_smooth.clamp(min=self.eps)

        self.v[:] = volume
        if self.use_dollar_volume:
            self.v[:] *= self.p
        self.v_smooth += alpha_p * (self.v[None, :] - self.v_smooth)
        self.v_rel = (self.v[None, :] - self.v_smooth) / self.v_smooth.clamp(min=self.eps)

    # -------------------------------------------------------------------------
    # trading
    # -------------------------------------------------------------------------

    def _trade(self, a: int | torch.Tensor | None) -> tuple[torch.Tensor, bool]:
        """Trade for MultiCurrencyEnv.

        Args:
            a (int | torch.Tensor | None): The a value.

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

        action_info = torch.tensor([0.0, -1.0, 1.0], dtype=self.dtype)

        if action_idx == 0:
            return action_info, True

        k = action_idx - 1
        valid_trade = False

        if self.w[k] > self.eps:
            # --------------------------------------------------
            # SELL ALL units of asset k
            # --------------------------------------------------
            action_info[0] = 2.0
            action_info[1] = float(k)

            units_to_sell = float(self.w[k].item())
            price_k = float(self.p[k].item())
            sell_value = units_to_sell * price_k

            if sell_value > self.s_fee:
                units_to_sell_t = self.w[k].clone()
                realized_cost = self.invested[k].clone()

                sell_value_t = units_to_sell_t * self.p[k]
                proceeds = (sell_value_t - self.s_fee).clamp(min=0.0)
                pre_tax_pnl = proceeds - realized_cost
                tax = pre_tax_pnl.clamp(min=0.0) * self.tax_rate
                net_proceeds = proceeds - tax
                realized_pnl = net_proceeds - realized_cost

                self.C += net_proceeds
                self.w[k] = torch.tensor(0.0, dtype=self.dtype)
                self.invested[k] = torch.tensor(0.0, dtype=self.dtype)

                self.realized_cost[k] = realized_cost
                self.realized_pnl[k] = realized_pnl
                valid_trade = True

        else:
            # --------------------------------------------------
            # BUY asset k with all available cash
            # --------------------------------------------------
            action_info[0] = 1.0
            action_info[1] = float(k)

            cash = float(self.C.item())
            buy_dollars = cash - self.b_fee

            if cash > self.b_fee and buy_dollars >= self.min_buy_dollars:
                price_k = self.p[k].clamp(min=self.eps)
                buy_dollars_t = torch.tensor(buy_dollars, dtype=self.dtype)
                fee_t = torch.tensor(self.b_fee, dtype=self.dtype)

                buy_units = buy_dollars_t / price_k
                total_cost = torch.minimum(buy_dollars_t + fee_t, self.C)

                self.C -= total_cost
                self.w[k] += buy_units
                self.invested[k] += total_cost
                valid_trade = True

        # dust cleanup
        dust = (self.w * self.p) < self.transaction_eps
        self.w[dust] = 0.0
        self.invested[dust] = 0.0
        self.C = torch.clamp(self.C, min=0.0)

        return action_info, valid_trade

    # -------------------------------------------------------------------------
    # reward
    # -------------------------------------------------------------------------

    def _reward(self) -> float:
        """Reward for MultiCurrencyEnv.

        Returns:
            float: The computed or requested result.
        """
        if self.realized_cost.sum() > self.eps:
            reward = self.roi_coeff * (self.realized_pnl.sum() / self.realized_cost.sum()).item()
        else:
            if self.reward_mode == "return":
                reward = self.val_coeff * ((self.V - self.V_prev) / self.V_prev).detach().item()
            else:
                reward = self.val_coeff * torch.log(self.V / self.V_prev).detach().item()

        return float(reward)

    # -------------------------------------------------------------------------
    # main step
    # -------------------------------------------------------------------------

    def step(self, a: int | torch.Tensor | None, data: dict) -> Tuple[State, float, bool, Dict]:
        """Advance the simulation by one step.

        Args:
            a (int | torch.Tensor | None): The a value.
            data (dict): The data value.

        Returns:
            Tuple[State, float, bool, Dict]: The computed or requested result.
        """
        action_info, valid_trade = self._trade(a)
        self._update(data)
        reward = self._reward()

        done = float(self.V) <= self.bankruptcy_threshold
        if done:
            reward -= self.done_reward_penalty

        info = {
            "t": self.t,
            "action_type": int(action_info[0].item()),   # 0=hold, 1=buy, 2=sell
            "action_asset": int(action_info[1].item()),  # -1 for hold
            "valid_trade": bool(valid_trade),
            "C": float(self.C),
            "V": float(self.V),
            "w": self.w.tolist(),
            "p": self.p.tolist(),
            "realized_cost": self.realized_cost.tolist(),
            "realized_pnl": self.realized_pnl.tolist(),
        }

        next_state = self._get_state()
        if self.save_history and self.history is not None:
            self.history.append(next_state)

        return next_state, reward, done, info


class BatchedMultiCurrencyEnv:
    """
    Fully-batched B-environment version of MultiCurrencyEnv (v00 — no size buckets).

    Action space and semantics are identical to MultiCurrencyEnv above.
    Market data is passed as (B, N) tensors per call.
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
        sell_fee: float = 1.0,
        buy_fee: float = 1.0,
        tax_rate: float = 0.26,
        min_buy_dollars: float = 10.0,
        val_coeff: float = 1.0,
        roi_coeff: float = 1.0,
        reward_mode: str = "log",
        done_reward_penalty: float = 1.0,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ):
        """Initialize the instance.

        Args:
            B (int): The b value.
            N (int): The n value.
            C0 (float): The c0 value.
            tau_p (torch.Tensor): The tau p value.
            bankruptcy_threshold (float): The bankruptcy threshold value. Defaults to ``1.0``.
            transaction_eps (float): The transaction eps value. Defaults to ``0.01``.
            use_dollar_volume (bool): The use dollar volume value. Defaults to ``True``.
            sell_fee (float): The sell fee value. Defaults to ``1.0``.
            buy_fee (float): The buy fee value. Defaults to ``1.0``.
            tax_rate (float): The tax rate value. Defaults to ``0.26``.
            min_buy_dollars (float): The min buy dollars value. Defaults to ``10.0``.
            val_coeff (float): The val coeff value. Defaults to ``1.0``.
            roi_coeff (float): The roi coeff value. Defaults to ``1.0``.
            reward_mode (str): The reward mode value. Defaults to ``'log'``.
            done_reward_penalty (float): The done reward penalty value. Defaults to ``1.0``.
            dtype (torch.dtype): The dtype value. Defaults to ``torch.float32``.
            eps (float): The eps value. Defaults to ``1e-08``.

        Returns:
            None: This function does not return a value.
        """
        assert isinstance(B, int) and B > 0
        assert isinstance(N, int) and N > 0
        assert tau_p.ndim == 1 and tau_p.numel() > 0
        assert min_buy_dollars >= buy_fee
        assert reward_mode in ("return", "log")

        self.B = B
        self.N = N
        self.dtype = dtype
        self.eps = float(eps)
        self.C0 = float(C0)
        self.bankruptcy_threshold = float(bankruptcy_threshold)
        self.transaction_eps = float(transaction_eps)
        self.use_dollar_volume = bool(use_dollar_volume)
        self.tax_rate = float(tax_rate)
        self.min_buy_dollars = float(min_buy_dollars)
        self.val_coeff = float(val_coeff)
        self.roi_coeff = float(roi_coeff)
        self.reward_mode = reward_mode
        self.done_reward_penalty = float(done_reward_penalty)
        self.s_fee = float(sell_fee)
        self.b_fee = float(buy_fee)

        self.tau_p = tau_p.to(dtype)        # (M,)
        self.M = self.tau_p.numel()

        # state = time(6) + p_rel(M*N) + hl_rel(N) + x_rel(N+1) + c_rel(N) + rho(1) + v_rel(M*N) + unrl_rel(N) + vol_rel(M*N)
        self.state_dim = 3 * self.M * N + 4 * N + 8
        self.action_dim = 1 + N

        self._b_idx = torch.arange(B)   # (B,) — reused in _trade for fancy indexing

        # runtime state — initialised in reset()
        self.C = None           # (B,)
        self.w = None           # (B, N)
        self.invested = None    # (B, N)
        self.p = None           # (B, N)
        self.v = None           # (B, N)
        self.p_smooth = None    # (B, M, N)
        self.v_smooth = None    # (B, M, N)
        self.hl_rel = None      # (B, N)
        self.hl_smooth = None   # (B, M, N)
        self.p_rel = None       # (B, M, N)
        self.v_rel = None       # (B, M, N)
        self.vol_rel = None     # (B, M, N)
        self.V_prev = None      # (B,)
        self.realized_cost = None   # (B, N)
        self.realized_pnl = None    # (B, N)
        self.t = None           # (B,) float timestamps

    # -------------------------------------------------------------------------
    # action helpers
    # -------------------------------------------------------------------------

    def encode_hold(self) -> int:
        """Encode hold for BatchedMultiCurrencyEnv.

        Returns:
            int: The computed or requested result.
        """
        return 0

    def encode_asset(self, asset_idx: int) -> int:
        """Encode asset for BatchedMultiCurrencyEnv.

        Args:
            asset_idx (int): The asset idx value.

        Returns:
            int: The computed or requested result.
        """
        return 1 + asset_idx

    # -------------------------------------------------------------------------
    # portfolio property
    # -------------------------------------------------------------------------

    @property
    def V(self) -> torch.Tensor:    # (B,)
        """V for BatchedMultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return self.C + (self.w * self.p).sum(dim=-1)

    # -------------------------------------------------------------------------
    # reset
    # -------------------------------------------------------------------------

    def reset(
        self,
        close: torch.Tensor,    # (B, N)
        high: torch.Tensor,     # (B, N)
        low: torch.Tensor,      # (B, N)
        volume: torch.Tensor,   # (B, N)
        time: torch.Tensor,     # (B,) float timestamps
        C0: Optional[float] = None,
    ) -> torch.Tensor:
        """Reset internal state for a new episode or stream.

        Args:
            close (torch.Tensor): The close value.
            high (torch.Tensor): The high value.
            low (torch.Tensor): The low value.
            volume (torch.Tensor): The volume value.
            time (torch.Tensor): The time value.
            C0 (Optional[float]): The c0 value. Defaults to ``None``.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        if C0 is not None:
            self.C0 = float(C0)

        B, N, M = self.B, self.N, self.M

        close  = close.to(self.dtype)
        high   = high.to(self.dtype)
        low    = low.to(self.dtype)
        volume = volume.to(self.dtype)
        self.t = time.to(torch.float64) if isinstance(time, torch.Tensor) else torch.tensor(time, dtype=torch.float64)

        self.C        = torch.full((B,), self.C0, dtype=self.dtype)
        self.w        = torch.zeros(B, N, dtype=self.dtype)
        self.invested = torch.zeros(B, N, dtype=self.dtype)
        self.p        = close.clone()
        self.v        = volume.clone()
        if self.use_dollar_volume:
            self.v = self.v * self.p

        self.p_smooth  = self.p[:, None, :].expand(B, M, N).clone()   # (B, M, N)
        self.v_smooth  = self.v[:, None, :].expand(B, M, N).clone()
        self.hl_rel    = self._hl_rel(high, low)                        # (B, N)
        self.hl_smooth = self.hl_rel[:, None, :].expand(B, M, N).clone()

        self.p_rel  = torch.zeros(B, M, N, dtype=self.dtype)
        self.v_rel  = torch.zeros(B, M, N, dtype=self.dtype)
        self.vol_rel = torch.zeros(B, M, N, dtype=self.dtype)

        self.V_prev       = self.C.clone()
        self.realized_cost = torch.zeros(B, N, dtype=self.dtype)
        self.realized_pnl  = torch.zeros(B, N, dtype=self.dtype)

        return self._obs()

    # -------------------------------------------------------------------------
    # internal helpers
    # -------------------------------------------------------------------------

    def _hl_rel(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        """Hl rel for BatchedMultiCurrencyEnv.

        Args:
            high (torch.Tensor): The high value.
            low (torch.Tensor): The low value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return 2.0 * (high - low) / (high + low + self.eps)    # (B, N)

    def _time_features(self) -> torch.Tensor:
        """Compute (B, 6) cyclic time features.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        features = []
        for ts in self.t.tolist():
            dt_obj = datetime.fromtimestamp(ts)
            iso = dt_obj.isocalendar()
            total_weeks = datetime(dt_obj.year, 12, 28).isocalendar().week
            period_day  = (dt_obj.hour + (dt_obj.minute + dt_obj.microsecond / 1e6) / 60.0) / 24.0
            period_week = (iso.weekday - 1 + period_day) / 7.0
            period_year = (iso.week - 1 + period_week) / max(total_weeks, 1)
            angles = torch.tensor([period_day, period_week, period_year], dtype=self.dtype)
            features.append(torch.cat([torch.sin(2 * PI * angles), torch.cos(2 * PI * angles)]))
        return torch.stack(features)    # (B, 6)

    def _obs(self) -> torch.Tensor:
        """Build and return the (B, state_dim) observation tensor.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        B, N = self.B, self.N
        V = self.V.clamp(min=self.eps)  # (B,)

        t_vec   = self._time_features()                             # (B, 6)
        x_cash  = (self.C / V)[:, None]                             # (B, 1)
        x_assets = (self.w * self.p) / V[:, None]                  # (B, N)
        x_rel   = torch.cat([x_cash, x_assets], dim=1)             # (B, N+1)

        S     = self.invested.sum(dim=1)                            # (B,)
        c_rel = torch.where(
            S[:, None] > 0,
            self.invested / (S[:, None] + self.eps),
            torch.zeros(B, N, dtype=self.dtype),
        )                                                           # (B, N)
        rho   = S / (S + self.C + self.eps)                         # (B,)

        has       = self.invested > self.eps                        # (B, N)
        avg_cost  = torch.where(has, self.invested / self.w.clamp(min=self.eps),
                                torch.zeros(B, N, dtype=self.dtype))
        unrl_pre  = torch.where(has, (self.p - avg_cost) * self.w,
                                torch.zeros(B, N, dtype=self.dtype))
        unrl_rel  = torch.where(
            has,
            (unrl_pre - unrl_pre.clamp(min=0.0) * self.tax_rate) / self.invested.clamp(min=self.eps),
            torch.zeros(B, N, dtype=self.dtype),
        )                                                           # (B, N)

        return torch.cat([
            t_vec,                          # (B, 6)
            self.p_rel.view(B, -1),         # (B, M*N)
            self.hl_rel,                    # (B, N)
            x_rel,                          # (B, N+1)
            c_rel,                          # (B, N)
            rho[:, None],                   # (B, 1)
            self.v_rel.view(B, -1),         # (B, M*N)
            unrl_rel,                       # (B, N)
            self.vol_rel.view(B, -1),       # (B, M*N)
        ], dim=1)

    def _update(
        self,
        close: torch.Tensor,    # (B, N)
        high: torch.Tensor,     # (B, N)
        low: torch.Tensor,      # (B, N)
        volume: torch.Tensor,   # (B, N)
        time: torch.Tensor,     # (B,)
    ) -> None:
        """Apply one update step.

        Args:
            close (torch.Tensor): The close value.
            high (torch.Tensor): The high value.
            low (torch.Tensor): The low value.
            volume (torch.Tensor): The volume value.
            time (torch.Tensor): The time value.

        Returns:
            None: This function does not return a value.
        """
        self.V_prev = self.V.detach().clone()   # (B,)

        close  = close.to(self.dtype)
        high   = high.to(self.dtype)
        low    = low.to(self.dtype)
        volume = volume.to(self.dtype)
        t_new  = time.to(torch.float64) if isinstance(time, torch.Tensor) else torch.tensor(time, dtype=torch.float64)

        dt = (t_new - self.t).clamp(min=0.0).to(self.dtype)    # (B,)
        self.t = t_new

        self.p = close

        alpha_p = 1.0 - torch.exp(
            -dt[:, None, None] / self.tau_p[None, :, None].clamp(min=self.eps)
        )

        p_exp = self.p[:, None, :]                                  # (B, 1, N)
        self.p_smooth = self.p_smooth + alpha_p * (p_exp - self.p_smooth)
        self.p_rel    = (p_exp - self.p_smooth) / self.p_smooth.clamp(min=self.eps)

        hl_new  = self._hl_rel(high, low)                           # (B, N)
        self.hl_rel = hl_new
        hl_exp  = hl_new[:, None, :]
        self.hl_smooth = self.hl_smooth + alpha_p * (hl_exp - self.hl_smooth)
        self.vol_rel   = (hl_exp - self.hl_smooth) / self.hl_smooth.clamp(min=self.eps)

        self.v = volume * self.p if self.use_dollar_volume else volume
        v_exp  = self.v[:, None, :]
        self.v_smooth = self.v_smooth + alpha_p * (v_exp - self.v_smooth)
        self.v_rel    = (v_exp - self.v_smooth) / self.v_smooth.clamp(min=self.eps)

    def _trade(self, actions: torch.Tensor) -> None:
        """Execute actions for all B environments in parallel.
        actions: (B,) int tensor.
        Updates C, w, invested, realized_cost, realized_pnl in-place.

        Args:
            actions (torch.Tensor): The actions value.

        Returns:
            None: This function does not return a value.
        """
        B, N = self.B, self.N
        b = self._b_idx   # (B,)

        self.realized_cost = torch.zeros(B, N, dtype=self.dtype)
        self.realized_pnl  = torch.zeros(B, N, dtype=self.dtype)

        is_toggle = actions >= 1                        # (B,)
        asset = (actions - 1).clamp(0, N - 1)          # (B,)

        # Determine whether each env buys or sells based on current position
        has_pos = self.w[b, asset] > self.eps           # (B,)
        is_sell = is_toggle & has_pos                   # (B,)
        is_buy  = is_toggle & ~has_pos                  # (B,)

        # ---- SELL ALL ----
        units_held  = self.w[b, asset]                          # (B,)
        sell_value  = units_held * self.p[b, asset]             # (B,)
        sell_valid  = is_sell & (sell_value > self.s_fee)       # (B,)

        rc          = self.invested[b, asset]                   # full realized cost  (B,)
        proceeds    = (sell_value - self.s_fee).clamp(min=0.0)
        pre_tax_pnl = proceeds - rc
        net_proceeds = proceeds - pre_tax_pnl.clamp(min=0.0) * self.tax_rate
        rpnl        = net_proceeds - rc

        sv = sell_valid.to(self.dtype)
        self.C += net_proceeds * sv
        self.w.scatter_add_(1, asset[:, None], -(units_held * sv)[:, None])
        self.invested.scatter_add_(1, asset[:, None], -(rc * sv)[:, None])
        self.realized_cost.scatter_add_(1, asset[:, None], (rc * sv)[:, None])
        self.realized_pnl.scatter_add_(1, asset[:, None], (rpnl * sv)[:, None])

        # ---- BUY ALL CASH ----
        # C may have been updated by sell; for is_buy envs it is unchanged (sell was inactive)
        buy_dollars = self.C - self.b_fee                       # (B,)
        buy_valid   = is_buy & (self.C > self.b_fee) & (buy_dollars >= self.min_buy_dollars)

        buy_price  = self.p[b, asset].clamp(min=self.eps)       # (B,)
        buy_units  = buy_dollars / buy_price                    # (B,)
        total_cost = self.C.clone()                             # spend all cash (= buy_dollars + b_fee)

        bv = buy_valid.to(self.dtype)
        self.C -= total_cost * bv
        self.w.scatter_add_(1, asset[:, None], (buy_units * bv)[:, None])
        self.invested.scatter_add_(1, asset[:, None], (total_cost * bv)[:, None])

        # dust cleanup
        dust = (self.w * self.p) < self.transaction_eps         # (B, N)
        self.w        = self.w.masked_fill(dust, 0.0)
        self.invested = self.invested.masked_fill(dust, 0.0)
        self.C        = self.C.clamp(min=0.0)

    def _reward(self) -> torch.Tensor:
        """Reward for BatchedMultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        has_realized = self.realized_cost.sum(dim=1) > self.eps     # (B,)
        roi = self.realized_pnl.sum(dim=1) / self.realized_cost.sum(dim=1).clamp(min=self.eps)
        V   = self.V
        if self.reward_mode == "return":
            baseline = self.val_coeff * (V - self.V_prev) / self.V_prev.clamp(min=self.eps)
        else:
            baseline = self.val_coeff * torch.log(V / self.V_prev.clamp(min=self.eps))
        return torch.where(has_realized, self.roi_coeff * roi, baseline)    # (B,)

    # -------------------------------------------------------------------------
    # public step / mask
    # -------------------------------------------------------------------------

    def step(
        self,
        actions: torch.Tensor,  # (B,) int
        close: torch.Tensor,    # (B, N)
        high: torch.Tensor,     # (B, N)
        low: torch.Tensor,      # (B, N)
        volume: torch.Tensor,   # (B, N)
        time: torch.Tensor,     # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns obs (B, state_dim), rewards (B,), dones (B,).

        Args:
            actions (torch.Tensor): The actions value.
            close (torch.Tensor): The close value.
            high (torch.Tensor): The high value.
            low (torch.Tensor): The low value.
            volume (torch.Tensor): The volume value.
            time (torch.Tensor): The time value.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The computed or requested result.
        """
        self._trade(actions)
        self._update(close, high, low, volume, time)
        rewards = self._reward()
        dones   = self.V <= self.bankruptcy_threshold               # (B,) bool
        rewards = torch.where(dones, rewards - self.done_reward_penalty, rewards)
        return self._obs(), rewards, dones

    def valid_action_mask(self) -> torch.Tensor:
        """Returns (B, action_dim) bool tensor.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        B, N = self.B, self.N
        mask = torch.zeros(B, self.action_dim, dtype=torch.bool)
        mask[:, 0] = True   # hold always valid

        can_buy = (self.C > self.b_fee) & ((self.C - self.b_fee) >= self.min_buy_dollars)  # (B,)

        for k in range(N):
            has_pos    = self.w[:, k] > self.eps                        # (B,)
            sell_valid = has_pos & (self.w[:, k] * self.p[:, k] > self.s_fee)
            mask[:, k + 1] = sell_valid | (~has_pos & can_buy)

        return mask
