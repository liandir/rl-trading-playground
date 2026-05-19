"""Discrete log utilities for trading environment state, action, reward, and simulation logic."""
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch

PI = 3.141592653589793238462


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class State:
    """State implementation for trading environment state, action, reward, and simulation logic."""
    time:      torch.Tensor   # [6]      cyclic time encoding
    log_ret:   torch.Tensor   # [N]      log(p_t / p_{t-1}) — per-step log-return
    close_loc: torch.Tensor   # [N]      (close - low) / (high - low) — close position within bar
    bar_ret:   torch.Tensor   # [N]      (close - open) / midprice — relative bar body (open = prev close)
    p_rel:     torch.Tensor   # [M, N]   log(p / EMA_p)     — price vs M trend EMAs
    log_hl:    torch.Tensor   # [N]      log(hl / EMA_hl_long) — current bar range vs long baseline
    vol_rel:   torch.Tensor   # [M, N]   (hl - EMA_hl) / EMA_hl — volatility regime across M scales
    v_rel:     torch.Tensor   # [M, N]   log(v / EMA_v)         — volume vs M trend EMAs
    x_rel:     torch.Tensor   # [N+1]    portfolio value weights (cash, assets)
    rho:       torch.Tensor   # []       overall invested fraction
    has_pos:   torch.Tensor   # [N]      binary: position is open
    unrl_rel:  torch.Tensor   # [N]      unrealized after-tax PnL / invested (0 if no position)

    def to_tensor(self) -> torch.Tensor:
        """Convert the value to tensor.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.cat([
            self.time.flatten(),
            self.log_ret.flatten(),
            self.close_loc.flatten(),
            self.bar_ret.flatten(),
            self.p_rel.flatten(),
            self.log_hl.flatten(),
            self.vol_rel.flatten(),
            self.v_rel.flatten(),
            self.x_rel.flatten(),
            self.rho.view(1),
            self.has_pos.flatten(),
            self.unrl_rel.flatten(),
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

    def to_tensor(self) -> torch.Tensor:
        """Convert the value to tensor.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.to_tensor() for s in self.states])


# ---------------------------------------------------------------------------
# MultiCurrencyEnv
# ---------------------------------------------------------------------------

class MultiCurrencyEnv:
    """
    Discrete-action multi-asset trading environment (v03).

    Combines the v02 feature set with the v00 toggle action space.

    Action space:
        0       -> hold
        1 .. N  -> toggle asset i:
                     if position open  -> sell all units of asset i
                     if no position    -> spend all cash on asset i

    State features:
        time      [6]      sin/cos encoding of time-of-day, day-of-week, week-of-year
        log_ret   [N]      log(p_t / p_{t-1})              — per-step log-return
        close_loc [N]      (close - low) / (high - low)    — close position within bar (0=low, 1=high)
        bar_ret   [N]      (close - open) / midprice       — relative bar body (open = prev close)
        p_rel     [M, N]   log(p / EMA_p)                  — price vs M trend EMAs (log-space)
        log_hl    [N]      log(hl_rel / EMA_hl_long)       — current bar range vs longest EMA baseline
        vol_rel   [M, N]   (hl_rel - EMA_hl) / EMA_hl     — volatility regime across M scales
        v_rel     [M, N]   log(v / EMA_v)                  — volume vs M trend EMAs (log-space)
        x_rel     [N+1]    cash + asset value fractions
        rho       [1]      invested / portfolio (overall exposure)
        has_pos   [N]      binary position indicator
        unrl_rel  [N]      unrealized after-tax PnL / invested cost (0 when flat)

    state_dim = 8 + 7*N + 3*M*N
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
        roi_coeff: float = 10.0,
        reward_mode: str = "log",
        done_reward_penalty: float = 10.0,
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
            roi_coeff (float): The roi coeff value. Defaults to ``10.0``.
            reward_mode (str): The reward mode value. Defaults to ``'log'``.
            done_reward_penalty (float): The done reward penalty value. Defaults to ``10.0``.
            dtype (torch.dtype): The dtype value. Defaults to ``torch.float32``.
            eps (float): The eps value. Defaults to ``1e-08``.

        Returns:
            None: This function does not return a value.
        """
        assert isinstance(N, int) and N > 0
        assert tau_p.ndim == 1 and tau_p.numel() > 0
        assert min_buy_dollars >= buy_fee
        assert reward_mode in ("return", "log")

        self.N    = N
        self.dtype = dtype
        self.eps  = float(eps)

        self.bankruptcy_threshold = float(bankruptcy_threshold)
        self.transaction_eps      = float(transaction_eps)
        self.use_dollar_volume    = bool(use_dollar_volume)
        self.tax_rate             = float(tax_rate)
        self.min_buy_dollars      = float(min_buy_dollars)
        self.val_coeff            = float(val_coeff)
        self.roi_coeff            = float(roi_coeff)
        self.reward_mode          = reward_mode
        self.done_reward_penalty  = float(done_reward_penalty)
        self.s_fee                = float(sell_fee)
        self.b_fee                = float(buy_fee)
        self.C0                   = float(C0)
        self.save_history         = bool(save_history)

        self.tau_p = tau_p.to(self.dtype)
        self.M     = self.tau_p.numel()

        self.state_dim  = 8 + 7 * N + 3 * self.M * N
        self.action_dim = 1 + N

        # runtime state — set in reset()
        self.C = self.w = self.t = self.dt = None
        self.p = self.p_prev = self.v = None
        self.p_smooth = self.p_rel = self.log_ret = None
        self.close_loc = self.bar_ret = None
        self.hl_rel = self.hl_smooth = self.vol_rel = self.log_hl = None
        self.v_smooth = self.v_rel = None
        self.V_prev = self.invested = None
        self.realized_cost = self.realized_pnl = None
        self.history = None

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
        """Valid action mask for MultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[0] = True

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
    # portfolio property
    # -------------------------------------------------------------------------

    @property
    def V(self) -> torch.Tensor:
        """V for MultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return self.C + torch.sum(self.w * self.p)

    # -------------------------------------------------------------------------
    # state construction helpers
    # -------------------------------------------------------------------------

    def _compute_time_vector(self) -> torch.Tensor:
        """Compute the time vector.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        dt_obj      = datetime.fromtimestamp(self.t)
        iso         = dt_obj.isocalendar()
        total_weeks = datetime(dt_obj.year, 12, 28).isocalendar().week
        period_day  = (dt_obj.hour + (dt_obj.minute + dt_obj.microsecond / 1e6) / 60.0) / 24.0
        period_week = (iso.weekday - 1 + period_day) / 7.0
        period_year = (iso.week - 1 + period_week) / max(total_weeks, 1)
        angles      = torch.tensor([period_day, period_week, period_year], dtype=self.dtype)
        return torch.cat([torch.sin(2 * PI * angles), torch.cos(2 * PI * angles)])

    def _compute_value_weights(self) -> torch.Tensor:
        """Compute the value weights.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        V = self.V.clamp(min=self.eps)
        return torch.cat([(self.C / V)[None], (self.w * self.p) / V])

    def _compute_rho(self) -> torch.Tensor:
        """Compute the rho.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        S = self.invested.sum()
        return S / (S + self.C + self.eps)

    def _compute_unrl_rel(self) -> torch.Tensor:
        """Compute the unrl rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        has      = self.invested > self.eps
        avg_cost = torch.zeros_like(self.invested)
        avg_cost[has] = self.invested[has] / self.w[has].clamp(min=self.eps)
        unrl_pre = torch.zeros(self.N, dtype=self.dtype)
        unrl_pre[has] = (self.p[has] - avg_cost[has]) * self.w[has]
        unrl_tax = unrl_pre.clamp(min=0.0) * self.tax_rate
        unrl_rel = torch.zeros(self.N, dtype=self.dtype)
        unrl_rel[has] = (unrl_pre[has] - unrl_tax[has]) / self.invested[has].clamp(min=self.eps)
        return unrl_rel

    def _get_state(self) -> State:
        """Return the current state snapshot.

        Returns:
            State: The computed or requested result.
        """
        has_pos = (self.w > self.eps).to(self.dtype)
        return State(
            time      = self._compute_time_vector(),
            log_ret   = self.log_ret.clone(),
            close_loc = self.close_loc.clone(),
            bar_ret   = self.bar_ret.clone(),
            p_rel     = self.p_rel.clone(),
            log_hl    = self.log_hl.clone(),
            vol_rel   = self.vol_rel.clone(),
            v_rel     = self.v_rel.clone(),
            x_rel     = self._compute_value_weights(),
            rho       = self._compute_rho(),
            has_pos   = has_pos,
            unrl_rel  = self._compute_unrl_rel(),
        )

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

        self.C        = torch.tensor(self.C0, dtype=self.dtype)
        self.w        = torch.zeros(self.N, dtype=self.dtype)
        self.invested = torch.zeros(self.N, dtype=self.dtype)
        self.realized_cost = torch.zeros(self.N, dtype=self.dtype)
        self.realized_pnl  = torch.zeros(self.N, dtype=self.dtype)

        t, close, high, low, volume = self._get_market_inputs(data)
        self.t = t
        self.dt = 0.0

        self.p      = close.clone()
        self.p_prev = close.clone()
        self.v      = volume.clone()
        if self.use_dollar_volume:
            self.v = self.v * self.p

        # EMAs initialised to current value → all relative features start at zero
        self.p_smooth  = self.p[None, :].repeat(self.M, 1).clone()     # (M, N)
        self.hl_rel    = self._hl_rel_single(high, low)                 # (N,)
        self.hl_smooth = self.hl_rel[None, :].repeat(self.M, 1).clone() # (M, N)
        self.v_smooth  = self.v[None, :].repeat(self.M, 1).clone()     # (M, N)

        self.log_ret   = torch.zeros(self.N, dtype=self.dtype)
        self.close_loc = torch.full((self.N,), 0.5, dtype=self.dtype)
        self.bar_ret   = torch.zeros(self.N, dtype=self.dtype)
        self.p_rel     = torch.zeros(self.M, self.N, dtype=self.dtype)
        self.log_hl    = torch.zeros(self.N, dtype=self.dtype)
        self.vol_rel   = torch.zeros(self.M, self.N, dtype=self.dtype)
        self.v_rel     = torch.zeros(self.M, self.N, dtype=self.dtype)

        self.V_prev = self.V.detach().clone()

        state = self._get_state()
        if self.save_history:
            self.history = StateHistory()
            self.history.append(state)
        return state

    def _hl_rel_single(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        """Hl rel single for MultiCurrencyEnv.

        Args:
            high (torch.Tensor): The high value.
            low (torch.Tensor): The low value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return 2.0 * (high - low) / (high + low + self.eps)

    def _get_market_inputs(self, data: dict):
        """Return the market inputs.

        Args:
            data (dict): The data value.

        Returns:
            Any: The computed or requested result.
        """
        for key in ("close", "high", "low", "volume"):
            if key not in data:
                raise KeyError(f"data must contain '{key}'")
        close  = data["close"].to(self.dtype).reshape(-1)
        high   = data["high"].to(self.dtype).reshape(-1)
        low    = data["low"].to(self.dtype).reshape(-1)
        volume = data["volume"].to(self.dtype).reshape(-1)
        for name, val in (("close", close), ("high", high), ("low", low), ("volume", volume)):
            if val.numel() != self.N:
                raise ValueError(f"data['{name}'] must have {self.N} elements")
        return float(data["time"]), close, high, low, volume

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
        self.t  = t_new

        # ---- price update ----
        self.p_prev = self.p.clone()
        self.p = close.clone()

        self.log_ret = torch.log((self.p / self.p_prev.clamp(min=self.eps)).clamp(min=self.eps))

        # ---- bar microstructure ----
        bar_range      = (high - low).clamp(min=self.eps)
        self.close_loc = (close - low) / bar_range
        mid            = (close + self.p_prev) * 0.5
        self.bar_ret   = (close - self.p_prev) / mid.clamp(min=self.eps)

        # ---- EMA decay ----
        alpha_p = (1.0 - torch.exp(-self.dt / self.tau_p.clamp(min=self.eps)))[:, None]  # (M, 1)

        self.p_smooth = self.p_smooth + alpha_p * (self.p[None, :] - self.p_smooth)
        self.p_rel    = torch.log((self.p[None, :] / self.p_smooth.clamp(min=self.eps)).clamp(min=self.eps))

        # ---- HL / volatility ----
        hl_new         = self._hl_rel_single(high, low)
        self.hl_rel    = hl_new
        self.hl_smooth = self.hl_smooth + alpha_p * (hl_new[None, :] - self.hl_smooth)
        self.vol_rel   = (hl_new[None, :] - self.hl_smooth) / self.hl_smooth.clamp(min=self.eps)
        # log of current bar range vs longest-timescale baseline
        self.log_hl = torch.log(
            (hl_new / self.hl_smooth[-1].clamp(min=self.eps)).clamp(min=self.eps)
        )

        # ---- volume ----
        self.v = volume * self.p if self.use_dollar_volume else volume.clone()
        self.v_smooth = self.v_smooth + alpha_p * (self.v[None, :] - self.v_smooth)
        self.v_rel = torch.log(
            (self.v[None, :] / self.v_smooth.clamp(min=self.eps)).clamp(min=self.eps)
        )

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

        action_idx = 0 if a is None else int(a.item() if isinstance(a, torch.Tensor) else a)
        if not (0 <= action_idx < self.action_dim):
            raise ValueError(f"Invalid action {action_idx}")

        action_info = torch.tensor([0.0, -1.0, 1.0], dtype=self.dtype)
        if action_idx == 0:
            return action_info, True

        k = action_idx - 1
        valid_trade = False

        if self.w[k] > self.eps:
            # SELL ALL units of asset k
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
            # BUY asset k with all available cash
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
            return float(
                self.roi_coeff * (self.realized_pnl.sum() / self.realized_cost.sum()).item()
            )
        if self.reward_mode == "return":
            return float(self.val_coeff * ((self.V - self.V_prev) / self.V_prev.clamp(min=self.eps)).item())
        return float(self.val_coeff * torch.log((self.V / self.V_prev.clamp(min=self.eps)).clamp(min=self.eps)).item())

    # -------------------------------------------------------------------------
    # step
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
            "t":              self.t,
            "action_type":    int(action_info[0].item()),
            "action_asset":   int(action_info[1].item()),
            "valid_trade":    bool(valid_trade),
            "C":              float(self.C),
            "V":              float(self.V),
            "w":              self.w.tolist(),
            "p":              self.p.tolist(),
            "realized_cost":  self.realized_cost.tolist(),
            "realized_pnl":   self.realized_pnl.tolist(),
        }

        next_state = self._get_state()
        if self.save_history and self.history is not None:
            self.history.append(next_state)
        return next_state, reward, done, info


# ---------------------------------------------------------------------------
# BatchedMultiCurrencyEnv
# ---------------------------------------------------------------------------

class BatchedMultiCurrencyEnv:
    """
    Fully-batched B-environment version (v02 features + toggle actions).

    All portfolio state is (B, ...) tensors — a single step() advances all B
    environments in parallel with no Python loops over environments.

    Action space:
        0       -> hold
        1 .. N  -> toggle asset i (sell all if open, buy all cash if flat)

    API
    ---
    reset(close, high, low, volume, time)         -> obs (B, state_dim)
    step(actions, close, high, low, volume, time) -> obs, rewards, dones

    state_dim = 8 + 7*N + 3*M*N
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
        roi_coeff: float = 10.0,
        reward_mode: str = "log",
        done_reward_penalty: float = 10.0,
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
            roi_coeff (float): The roi coeff value. Defaults to ``10.0``.
            reward_mode (str): The reward mode value. Defaults to ``'log'``.
            done_reward_penalty (float): The done reward penalty value. Defaults to ``10.0``.
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

        self.B    = B
        self.N    = N
        self.dtype = dtype
        self.eps  = float(eps)
        self.C0   = float(C0)

        self.bankruptcy_threshold = float(bankruptcy_threshold)
        self.transaction_eps      = float(transaction_eps)
        self.use_dollar_volume    = bool(use_dollar_volume)
        self.tax_rate             = float(tax_rate)
        self.min_buy_dollars      = float(min_buy_dollars)
        self.val_coeff            = float(val_coeff)
        self.roi_coeff            = float(roi_coeff)
        self.reward_mode          = reward_mode
        self.done_reward_penalty  = float(done_reward_penalty)
        self.s_fee                = float(sell_fee)
        self.b_fee                = float(buy_fee)

        self.tau_p = tau_p.to(dtype)
        self.M     = self.tau_p.numel()

        self.state_dim  = 8 + 7 * N + 3 * self.M * N
        self.action_dim = 1 + N

        self._b_idx = torch.arange(B)

        # runtime state — set in reset()
        self.C = self.w = self.invested = None
        self.p = self.p_prev = self.v = None
        self.p_smooth = self.hl_rel = self.hl_smooth = self.v_smooth = None
        self.p_rel = self.log_ret = self.log_hl = self.vol_rel = self.v_rel = None
        self.close_loc = self.bar_ret = None
        self.V_prev = self.realized_cost = self.realized_pnl = None
        self.t = None

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
    def V(self) -> torch.Tensor:   # (B,)
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
        close:  torch.Tensor,   # (B, N)
        high:   torch.Tensor,   # (B, N)
        low:    torch.Tensor,   # (B, N)
        volume: torch.Tensor,   # (B, N)
        time:   torch.Tensor,   # (B,)
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
        self.realized_cost = torch.zeros(B, N, dtype=self.dtype)
        self.realized_pnl  = torch.zeros(B, N, dtype=self.dtype)

        self.p      = close.clone()
        self.p_prev = close.clone()
        self.v      = volume.clone()
        if self.use_dollar_volume:
            self.v = self.v * self.p

        hl = self._hl_rel(high, low)                               # (B, N)
        self.hl_rel    = hl
        self.p_smooth  = self.p[:, None, :].expand(B, M, N).clone()   # (B, M, N)
        self.hl_smooth = hl[:, None, :].expand(B, M, N).clone()
        self.v_smooth  = self.v[:, None, :].expand(B, M, N).clone()

        self.log_ret   = torch.zeros(B, N, dtype=self.dtype)
        self.close_loc = torch.full((B, N), 0.5, dtype=self.dtype)
        self.bar_ret   = torch.zeros(B, N, dtype=self.dtype)
        self.p_rel     = torch.zeros(B, M, N, dtype=self.dtype)
        self.log_hl    = torch.zeros(B, N, dtype=self.dtype)
        self.vol_rel   = torch.zeros(B, M, N, dtype=self.dtype)
        self.v_rel     = torch.zeros(B, M, N, dtype=self.dtype)

        self.V_prev = self.C.clone()
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
        return 2.0 * (high - low) / (high + low + self.eps)

    def _time_features(self) -> torch.Tensor:
        """(B, 6) cyclic time features.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        features = []
        for ts in self.t.tolist():
            dt_obj      = datetime.fromtimestamp(ts)
            iso         = dt_obj.isocalendar()
            total_weeks = datetime(dt_obj.year, 12, 28).isocalendar().week
            period_day  = (dt_obj.hour + (dt_obj.minute + dt_obj.microsecond / 1e6) / 60.0) / 24.0
            period_week = (iso.weekday - 1 + period_day) / 7.0
            period_year = (iso.week - 1 + period_week) / max(total_weeks, 1)
            angles = torch.tensor([period_day, period_week, period_year], dtype=self.dtype)
            features.append(torch.cat([torch.sin(2 * PI * angles), torch.cos(2 * PI * angles)]))
        return torch.stack(features)   # (B, 6)

    def _obs(self) -> torch.Tensor:
        """Obs for BatchedMultiCurrencyEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        B, N = self.B, self.N
        V    = self.V.clamp(min=self.eps)          # (B,)

        t_vec    = self._time_features()           # (B, 6)

        x_cash   = (self.C / V)[:, None]          # (B, 1)
        x_assets = (self.w * self.p) / V[:, None] # (B, N)
        x_rel    = torch.cat([x_cash, x_assets], dim=1)  # (B, N+1)

        S   = self.invested.sum(dim=1)             # (B,)
        rho = S / (S + self.C + self.eps)          # (B,)

        has      = self.invested > self.eps        # (B, N)
        avg_cost = torch.where(has, self.invested / self.w.clamp(min=self.eps),
                               torch.zeros(B, N, dtype=self.dtype))
        unrl_pre = torch.where(has, (self.p - avg_cost) * self.w,
                               torch.zeros(B, N, dtype=self.dtype))
        unrl_rel = torch.where(
            has,
            (unrl_pre - unrl_pre.clamp(min=0.0) * self.tax_rate) / self.invested.clamp(min=self.eps),
            torch.zeros(B, N, dtype=self.dtype),
        )                                          # (B, N)

        has_pos = has.to(self.dtype)               # (B, N)

        return torch.cat([
            t_vec,                                 # (B, 6)
            self.log_ret,                          # (B, N)
            self.close_loc,                        # (B, N)
            self.bar_ret,                          # (B, N)
            self.p_rel.view(B, -1),                # (B, M*N)
            self.log_hl,                           # (B, N)
            self.vol_rel.view(B, -1),              # (B, M*N)
            self.v_rel.view(B, -1),                # (B, M*N)
            x_rel,                                 # (B, N+1)
            rho[:, None],                          # (B, 1)
            has_pos,                               # (B, N)
            unrl_rel,                              # (B, N)
        ], dim=1)

    def _update(
        self,
        close:  torch.Tensor,   # (B, N)
        high:   torch.Tensor,   # (B, N)
        low:    torch.Tensor,   # (B, N)
        volume: torch.Tensor,   # (B, N)
        time:   torch.Tensor,   # (B,)
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
        self.V_prev = self.V.detach().clone()

        close  = close.to(self.dtype)
        high   = high.to(self.dtype)
        low    = low.to(self.dtype)
        volume = volume.to(self.dtype)
        t_new  = time.to(torch.float64) if isinstance(time, torch.Tensor) else torch.tensor(time, dtype=torch.float64)

        dt     = (t_new - self.t).clamp(min=0.0).to(self.dtype)   # (B,)
        self.t = t_new

        # ---- price update ----
        self.p_prev = self.p.clone()
        self.p = close.clone()

        self.log_ret = torch.log(
            (self.p / self.p_prev.clamp(min=self.eps)).clamp(min=self.eps)
        )

        # ---- bar microstructure ----
        bar_range      = (high - low).clamp(min=self.eps)       # (B, N)
        self.close_loc = (close - low) / bar_range              # (B, N)
        mid            = (close + self.p_prev) * 0.5            # (B, N)
        self.bar_ret   = (close - self.p_prev) / mid.clamp(min=self.eps)  # (B, N)

        # alpha_p: (B, M, 1)
        alpha_p = 1.0 - torch.exp(
            -dt[:, None, None] / self.tau_p[None, :, None].clamp(min=self.eps)
        )

        p_exp          = self.p[:, None, :]                        # (B, 1, N)
        self.p_smooth  = self.p_smooth + alpha_p * (p_exp - self.p_smooth)
        self.p_rel     = torch.log(
            (p_exp / self.p_smooth.clamp(min=self.eps)).clamp(min=self.eps)
        )

        hl_new         = self._hl_rel(high, low)                   # (B, N)
        self.hl_rel    = hl_new
        hl_exp         = hl_new[:, None, :]                        # (B, 1, N)
        self.hl_smooth = self.hl_smooth + alpha_p * (hl_exp - self.hl_smooth)
        self.vol_rel   = (hl_exp - self.hl_smooth) / self.hl_smooth.clamp(min=self.eps)
        # log of current bar range vs longest-timescale baseline  (B, M, N)[-1] = (B, N)
        self.log_hl    = torch.log(
            (hl_new / self.hl_smooth[:, -1, :].clamp(min=self.eps)).clamp(min=self.eps)
        )

        self.v        = volume * self.p if self.use_dollar_volume else volume.clone()
        v_exp          = self.v[:, None, :]
        self.v_smooth  = self.v_smooth + alpha_p * (v_exp - self.v_smooth)
        self.v_rel     = torch.log(
            (v_exp / self.v_smooth.clamp(min=self.eps)).clamp(min=self.eps)
        )

    def _trade(self, actions: torch.Tensor) -> None:
        """Trade for BatchedMultiCurrencyEnv.

        Args:
            actions (torch.Tensor): The actions value.

        Returns:
            None: This function does not return a value.
        """
        B, N = self.B, self.N
        b = self._b_idx

        self.realized_cost = torch.zeros(B, N, dtype=self.dtype)
        self.realized_pnl  = torch.zeros(B, N, dtype=self.dtype)

        is_toggle = actions >= 1                        # (B,)
        asset = (actions - 1).clamp(0, N - 1)          # (B,)

        has_pos = self.w[b, asset] > self.eps           # (B,)
        is_sell = is_toggle & has_pos                   # (B,)
        is_buy  = is_toggle & ~has_pos                  # (B,)

        # ---- SELL ALL ----
        units_held  = self.w[b, asset]                          # (B,)
        sell_value  = units_held * self.p[b, asset]             # (B,)
        sell_valid  = is_sell & (sell_value > self.s_fee)       # (B,)

        rc          = self.invested[b, asset]                   # (B,)
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
        buy_dollars = self.C - self.b_fee                       # (B,)
        buy_valid   = is_buy & (self.C > self.b_fee) & (buy_dollars >= self.min_buy_dollars)

        buy_price  = self.p[b, asset].clamp(min=self.eps)       # (B,)
        buy_units  = buy_dollars / buy_price                    # (B,)
        total_cost = self.C.clone()                             # spend all cash

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
        has_realized = self.realized_cost.sum(dim=1) > self.eps
        roi = self.realized_pnl.sum(dim=1) / self.realized_cost.sum(dim=1).clamp(min=self.eps)
        V   = self.V
        if self.reward_mode == "return":
            baseline = self.val_coeff * (V - self.V_prev) / self.V_prev.clamp(min=self.eps)
        else:
            baseline = self.val_coeff * torch.log((V / self.V_prev.clamp(min=self.eps)).clamp(min=self.eps))
        return torch.where(has_realized, self.roi_coeff * roi, baseline)

    def step(
        self,
        actions: torch.Tensor,  # (B,)
        close:   torch.Tensor,  # (B, N)
        high:    torch.Tensor,  # (B, N)
        low:     torch.Tensor,  # (B, N)
        volume:  torch.Tensor,  # (B, N)
        time:    torch.Tensor,  # (B,)
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
        dones   = self.V <= self.bankruptcy_threshold
        rewards = torch.where(dones, rewards - self.done_reward_penalty, rewards)
        return self._obs(), rewards, dones

    def valid_action_mask(self) -> torch.Tensor:
        """Returns (B, action_dim) bool tensor.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        B, N = self.B, self.N
        mask = torch.zeros(B, self.action_dim, dtype=torch.bool)
        mask[:, 0] = True

        can_buy = (self.C > self.b_fee) & ((self.C - self.b_fee) >= self.min_buy_dollars)

        for k in range(N):
            has_pos    = self.w[:, k] > self.eps
            sell_valid = has_pos & (self.w[:, k] * self.p[:, k] > self.s_fee)
            mask[:, k + 1] = sell_valid | (~has_pos & can_buy)

        return mask
