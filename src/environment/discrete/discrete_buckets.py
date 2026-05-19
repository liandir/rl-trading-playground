"""Discrete buckets utilities for trading environment state, action, reward, and simulation logic."""
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
    Discrete-action multi-asset trading environment.

    Action space:
        0                                   -> hold
        1 .. N*K                            -> buy  asset i with bucket k
        N*K+1 .. 2*N*K                      -> sell asset i with bucket k

    where:
        K = len(size_buckets)
        size_buckets default to (0.50, 1.00)

    Buy semantics:
        spend bucket_fraction * current cash on selected asset,
        subject to fixed buy fee and minimum buy dollars.

    Sell semantics:
        sell bucket_fraction * current units of selected asset,
        subject to fixed sell fee and tax on realized gains.

    Invalid trades:
        Actions that cannot be executed (too little cash, no position,
        proceeds <= fee, below minimum buy size) are silently ignored —
        use valid_action_mask() to filter logits before sampling.

    Episode termination:
        If portfolio value falls to the bankruptcy threshold or below, the episode
        ends and a configurable penalty is applied.
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
        size_buckets: Tuple[float, ...] = (0.50, 1.00),
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
            size_buckets (Tuple[float, ...]): The size buckets value. Defaults to ``(0.5, 1.0)``.
            done_reward_penalty (float): The done reward penalty value. Defaults to ``1.0``.
            dtype (torch.dtype): The dtype value. Defaults to ``torch.float32``.
            eps (float): The eps value. Defaults to ``1e-08``.

        Returns:
            None: This function does not return a value.
        """
        assert isinstance(N, int) and N > 0
        assert tau_p.ndim == 1 and tau_p.numel() > 0
        assert min_buy_dollars >= buy_fee, "min_buy_dollars must be >= buy_fee"
        assert len(size_buckets) > 0, "size_buckets must not be empty"
        assert all(0.0 < x <= 1.0 for x in size_buckets), "all size buckets must be in (0, 1]"
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

        self.size_buckets = torch.tensor(size_buckets, dtype=self.dtype)
        self.K = int(self.size_buckets.numel())

        # state = time(6) + p_rel(M*N) + hl_rel(N) + x_rel(N+1) + c_rel(N) + rho(1) + v_rel(M*N) + unrl_rel(N) + vol_rel(M*N)
        self.state_dim = 3 * self.M * N + 4 * N + 8

        # discrete actions: hold + buys + sells
        self.action_dim = 1 + 2 * self.N * self.K

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

    def encode_buy(self, asset_idx: int, bucket_idx: int) -> int:
        """Encode buy for MultiCurrencyEnv.

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

    def encode_sell(self, asset_idx: int, bucket_idx: int) -> int:
        """Encode sell for MultiCurrencyEnv.

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

    def decode_action(self, action_idx: int) -> tuple[str, int | None, float | None]:
        """Decode action for MultiCurrencyEnv.

        Args:
            action_idx (int): The action idx value.

        Returns:
            tuple[str, int | None, float | None]: The computed or requested result.
        """
        if action_idx == 0:
            return ("hold", None, None)

        buy_block = self.N * self.K
        sell_offset = 1 + buy_block

        if 1 <= action_idx < sell_offset:
            z = action_idx - 1
            asset_idx = z // self.K
            bucket_idx = z % self.K
            return ("buy", asset_idx, float(self.size_buckets[bucket_idx].item()))

        if sell_offset <= action_idx < self.action_dim:
            z = action_idx - sell_offset
            asset_idx = z // self.K
            bucket_idx = z % self.K
            return ("sell", asset_idx, float(self.size_buckets[bucket_idx].item()))

        raise ValueError(f"invalid action index {action_idx}")

    def valid_action_mask(self) -> torch.Tensor:
        """Boolean mask of shape [action_dim]. True = action is executable given
        current cash and positions. Apply to logits (set False positions to -inf)
        before sampling to avoid wasted steps on invalid actions.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[0] = True  # hold is always valid

        cash = float(self.C.item())

        # Buy validity depends only on bucket fraction and available cash,
        # not on which asset is being bought.
        for b_idx in range(self.K):
            frac = float(self.size_buckets[b_idx].item())
            budget = frac * cash
            if budget > self.b_fee and (budget - self.b_fee) >= self.min_buy_dollars:
                for k in range(self.N):
                    mask[self.encode_buy(k, b_idx)] = True

        # Sell validity is per-asset and per-bucket.
        for k in range(self.N):
            units = float(self.w[k].item())
            if units > 0.0:
                price = float(self.p[k].item())
                for b_idx in range(self.K):
                    frac = float(self.size_buckets[b_idx].item())
                    if frac * units * price > self.s_fee:
                        mask[self.encode_sell(k, b_idx)] = True

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

    def _compute_moneyness(self) -> torch.Tensor:
        """Compute the moneyness.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        has_pos = self.w > self.eps
        avg_cost = torch.zeros_like(self.invested)
        avg_cost[has_pos] = self.invested[has_pos] / self.w[has_pos].clamp(min=self.eps)

        m = torch.zeros(self.N, dtype=self.dtype)
        m[has_pos] = (self.p[has_pos] - avg_cost[has_pos]) / (avg_cost[has_pos] + self.eps)
        return m

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

        # dollars spent on each current position, including buy fees
        self.invested = torch.zeros(self.N, dtype=self.dtype)

        # realized bookkeeping per step
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
        """Documented callable.

        Args:
            a (int | torch.Tensor | None): The a value.

        Returns:
            tuple[torch.Tensor, bool]: The computed or requested result.
        """
        self.realized_cost.zero_()
        self.realized_pnl.zero_()

        # decode action index
        if a is None:
            action_idx = 0
        elif isinstance(a, torch.Tensor):
            action_idx = int(a.item())
        else:
            action_idx = int(a)

        if not (0 <= action_idx < self.action_dim):
            raise ValueError(f"Invalid discrete action {action_idx}, expected in [0, {self.action_dim - 1}]")

        action_info = torch.tensor([0.0, -1.0, 0.0], dtype=self.dtype)

        # hold
        if action_idx == 0:
            return action_info, True

        buy_block = self.N * self.K
        sell_offset = 1 + buy_block
        valid_trade = False

        # --------------------------------------------------
        # BUY: spend bucket fraction of cash on one asset
        # --------------------------------------------------
        if 1 <= action_idx < sell_offset:
            z = action_idx - 1
            k = z // self.K
            bucket_idx = z % self.K
            frac = float(self.size_buckets[bucket_idx].item())

            action_info[0] = 1.0      # buy
            action_info[1] = float(k) # asset index
            action_info[2] = frac     # bucket fraction

            cash = float(self.C.item())
            budget = frac * cash
            buy_dollars = budget - self.b_fee

            # economic validity only
            if budget > self.b_fee and buy_dollars >= self.min_buy_dollars:
                price_k = self.p[k].clamp(min=self.eps)
                buy_dollars_t = torch.tensor(buy_dollars, dtype=self.dtype)
                fee_t = torch.tensor(self.b_fee, dtype=self.dtype)

                buy_units = buy_dollars_t / price_k
                total_cost = buy_dollars_t + fee_t

                # final safety clamp against tiny float overshoot
                total_cost = torch.minimum(total_cost, self.C)

                self.C -= total_cost
                self.w[k] += buy_units
                self.invested[k] += total_cost
                valid_trade = True

        # --------------------------------------------------
        # SELL: sell bucket fraction of held units of one asset
        # --------------------------------------------------
        else:
            z = action_idx - sell_offset
            k = z // self.K
            bucket_idx = z % self.K
            frac = float(self.size_buckets[bucket_idx].item())

            action_info[0] = 2.0
            action_info[1] = float(k)
            action_info[2] = frac

            units_before = float(self.w[k].item())

            # economic validity only
            if units_before > 0.0:
                units_to_sell = frac * units_before
                price_k = float(self.p[k].item())
                sell_value = units_to_sell * price_k

                if sell_value > self.s_fee:
                    units_to_sell_t = torch.tensor(units_to_sell, dtype=self.dtype)
                    units_before_t = self.w[k].clamp(min=self.eps)

                    position_frac = (units_to_sell_t / units_before_t).clamp(0.0, 1.0)
                    realized_cost = position_frac * self.invested[k]

                    sell_value_t = units_to_sell_t * self.p[k]
                    proceeds = (sell_value_t - self.s_fee).clamp(min=0.0)
                    pre_tax_pnl = proceeds - realized_cost
                    tax = pre_tax_pnl.clamp(min=0.0) * self.tax_rate

                    net_proceeds = proceeds - tax
                    realized_pnl = net_proceeds - realized_cost

                    self.C += net_proceeds
                    self.w[k] -= units_to_sell_t
                    self.invested[k] -= realized_cost

                    self.realized_cost[k] = realized_cost
                    self.realized_pnl[k] = realized_pnl
                    valid_trade = True

        # dust cleanup remains an economic threshold
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
            "action_frac": float(action_info[2].item()),
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
    Fully-batched B-environment version of MultiCurrencyEnv.

    All portfolio state is stored as (B, ...) tensors so a single step() call
    advances all B environments in parallel using vectorised PyTorch operations
    — no Python loops over environments.

    Market data is passed as (B, N) tensors per call so each environment can
    independently track a different segment of history (different start offsets).

    Action space and parameters are identical to MultiCurrencyEnv.

    API
    ---
    reset(close, high, low, volume, time)               -> obs (B, state_dim)
    step(actions, close, high, low, volume, time)       -> obs, rewards, dones
    valid_action_mask()                                 -> (B, action_dim) bool
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
        size_buckets: Tuple[float, ...] = (0.50, 1.00),
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
            size_buckets (Tuple[float, ...]): The size buckets value. Defaults to ``(0.5, 1.0)``.
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
        assert len(size_buckets) > 0
        assert all(0.0 < x <= 1.0 for x in size_buckets)
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

        self.size_buckets = torch.tensor(size_buckets, dtype=dtype)  # (K,)
        self.K = int(self.size_buckets.numel())

        # state = time(6) + p_rel(M*N) + hl_rel(N) + x_rel(N+1) + c_rel(N) + rho(1) + v_rel(M*N) + unrl_rel(N) + vol_rel(M*N)
        self.state_dim = 3 * self.M * N + 4 * N + 8
        self.action_dim = 1 + 2 * N * self.K

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

    def encode_buy(self, asset_idx: int, bucket_idx: int) -> int:
        """Encode buy for BatchedMultiCurrencyEnv.

        Args:
            asset_idx (int): The asset idx value.
            bucket_idx (int): The bucket idx value.

        Returns:
            int: The computed or requested result.
        """
        return 1 + asset_idx * self.K + bucket_idx

    def encode_sell(self, asset_idx: int, bucket_idx: int) -> int:
        """Encode sell for BatchedMultiCurrencyEnv.

        Args:
            asset_idx (int): The asset idx value.
            bucket_idx (int): The bucket idx value.

        Returns:
            int: The computed or requested result.
        """
        return 1 + self.N * self.K + asset_idx * self.K + bucket_idx

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
        """Compute (B, 6) cyclic time features. Small Python loop over B is fine.

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

        dt = (t_new - self.t).clamp(min=0.0).to(self.dtype)    # (B,) diff in f64 → convert
        self.t = t_new

        self.p = close

        # alpha_p: (B, M, 1) — broadcasts over the N dimension
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
        B, N, K = self.B, self.N, self.K
        b = self._b_idx   # (B,)

        self.realized_cost = torch.zeros(B, N, dtype=self.dtype)
        self.realized_pnl  = torch.zeros(B, N, dtype=self.dtype)

        sell_offset = 1 + N * K
        is_buy  = (actions >= 1) & (actions < sell_offset)     # (B,)
        is_sell = actions >= sell_offset                        # (B,)

        # ---- BUY ----
        buy_z      = (actions - 1).clamp(min=0)                # (B,)
        buy_asset  = (buy_z // K).clamp(0, N - 1)              # (B,)
        buy_bucket = buy_z % K                                  # (B,)

        buy_frac   = self.size_buckets[buy_bucket]              # (B,)
        buy_budget = buy_frac * self.C                          # (B,)
        buy_dollars = buy_budget - self.b_fee                   # (B,)
        buy_valid  = is_buy & (buy_budget > self.b_fee) & (buy_dollars >= self.min_buy_dollars)

        buy_price  = self.p[b, buy_asset].clamp(min=self.eps)  # (B,)
        buy_units  = buy_dollars / buy_price                    # (B,)
        total_cost = torch.minimum(buy_budget, self.C)          # (B,) safety clamp

        bv = buy_valid.to(self.dtype)
        self.C -= total_cost * bv
        self.w.scatter_add_(1, buy_asset[:, None], (buy_units * bv)[:, None])
        self.invested.scatter_add_(1, buy_asset[:, None], (total_cost * bv)[:, None])

        # ---- SELL ----
        sell_z      = (actions - sell_offset).clamp(min=0)     # (B,)
        sell_asset  = (sell_z // K).clamp(0, N - 1)            # (B,)
        sell_bucket = sell_z % K                               # (B,)

        sell_frac   = self.size_buckets[sell_bucket]            # (B,)
        units_held  = self.w[b, sell_asset]                     # (B,)
        units_to_sell = sell_frac * units_held                  # (B,)
        sell_value  = units_to_sell * self.p[b, sell_asset]    # (B,)
        sell_valid  = is_sell & (units_held > 0.0) & (sell_value > self.s_fee)

        pos_frac    = (units_to_sell / units_held.clamp(min=self.eps)).clamp(0.0, 1.0)
        rc          = pos_frac * self.invested[b, sell_asset]   # realized cost  (B,)
        proceeds    = (sell_value - self.s_fee).clamp(min=0.0)
        pre_tax_pnl = proceeds - rc
        net_proceeds = proceeds - pre_tax_pnl.clamp(min=0.0) * self.tax_rate
        rpnl        = net_proceeds - rc

        sv = sell_valid.to(self.dtype)
        self.C += net_proceeds * sv
        self.w.scatter_add_(1, sell_asset[:, None], -(units_to_sell * sv)[:, None])
        self.invested.scatter_add_(1, sell_asset[:, None], -(rc * sv)[:, None])
        self.realized_cost.scatter_add_(1, sell_asset[:, None], (rc * sv)[:, None])
        self.realized_pnl.scatter_add_(1, sell_asset[:, None], (rpnl * sv)[:, None])

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
        B, N, K = self.B, self.N, self.K
        mask = torch.zeros(B, self.action_dim, dtype=torch.bool)
        mask[:, 0] = True   # hold always valid

        for b_idx in range(K):
            frac      = float(self.size_buckets[b_idx].item())
            budget    = frac * self.C                               # (B,)
            buy_valid = (budget > self.b_fee) & ((budget - self.b_fee) >= self.min_buy_dollars)
            for k in range(N):
                mask[:, self.encode_buy(k, b_idx)] = buy_valid

        for k in range(N):
            units = self.w[:, k]
            price = self.p[:, k]
            for b_idx in range(K):
                frac = float(self.size_buckets[b_idx].item())
                mask[:, self.encode_sell(k, b_idx)] = (units > 0.0) & (frac * units * price > self.s_fee)

        return mask
