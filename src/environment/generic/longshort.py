from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch

PI = 3.141592653589793238462


@dataclass
class State:
    """
    Hybrid long/short trading state.

    Flattened layout of `to_tensor()`:
        [ globals(8) | asset_0(4M+9) | asset_1(4M+9) | ... | asset_{N-1}(4M+9) ]

    Globals (8):
        time(6), cash_rel(1), rho(1)

    Per-asset block of size 4M+9:
        p_rel[:, k]          (M)
        v_rel[:, k]          (M)
        vol_rel[:, k]        (M)
        body_smooth[:, k]    (M)   EMA of (close-open)/range across tau scales
        hl_rel[k]            (1)
        body_rel[k]          (1)   signed (close-open)/range, in [-1, 1]
        ret_rel[k]           (1)   bar return (close-open)/open
        upper_wick_rel[k]    (1)   (high - max(open,close)) / range
        lower_wick_rel[k]    (1)   (min(open,close) - low) / range
        x_rel[k]             (1)   signed value weight of asset k
        c_rel[k]             (1)
        unrl_rel[k]          (1)
        side_rel[k]          (1)
    """

    # globals
    time: torch.Tensor            # [6]    cyclic time features
    cash_rel: torch.Tensor        # []     cash / V
    rho: torch.Tensor             # []     committed_sum / (committed_sum + cash)

    # per-asset
    p_rel: torch.Tensor           # [M, N]
    v_rel: torch.Tensor           # [M, N]
    vol_rel: torch.Tensor         # [M, N]
    body_smooth: torch.Tensor     # [M, N]
    hl_rel: torch.Tensor          # [N]
    body_rel: torch.Tensor        # [N]
    ret_rel: torch.Tensor         # [N]
    upper_wick_rel: torch.Tensor  # [N]
    lower_wick_rel: torch.Tensor  # [N]
    x_rel: torch.Tensor           # [N]    signed per-asset value weights (pos_units * p / V)
    c_rel: torch.Tensor           # [N]    committed distribution (>= 0)
    unrl_rel: torch.Tensor        # [N]    signed after-tax PnL / committed (0 when flat)
    side_rel: torch.Tensor        # [N]    -1 short, 0 flat, +1 long

    def to_tensor(self) -> torch.Tensor:
        globals_block = torch.cat([
            self.time.flatten(),
            self.cash_rel.reshape(1),
            self.rho.reshape(1),
        ])
        # (4M, N) -> (N, 4M)
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
        ], dim=1)  # (N, 9)
        asset_block = torch.cat([mn, scalars], dim=1)  # (N, 4M+9)
        return torch.cat([globals_block, asset_block.flatten()])


class StateHistory:
    def __init__(self):
        self.states: list[State] = []

    def append(self, state: State):
        self.states.append(state)

    def get_time(self) -> torch.Tensor:
        return torch.stack([s.time for s in self.states])

    def get_cash_rel(self) -> torch.Tensor:
        return torch.stack([s.cash_rel for s in self.states])

    def get_rho(self) -> torch.Tensor:
        return torch.stack([s.rho for s in self.states])

    def get_p_rel(self) -> torch.Tensor:
        return torch.stack([s.p_rel for s in self.states])

    def get_v_rel(self) -> torch.Tensor:
        return torch.stack([s.v_rel for s in self.states])

    def get_vol_rel(self) -> torch.Tensor:
        return torch.stack([s.vol_rel for s in self.states])

    def get_body_smooth(self) -> torch.Tensor:
        return torch.stack([s.body_smooth for s in self.states])

    def get_hl_rel(self) -> torch.Tensor:
        return torch.stack([s.hl_rel for s in self.states])

    def get_body_rel(self) -> torch.Tensor:
        return torch.stack([s.body_rel for s in self.states])

    def get_ret_rel(self) -> torch.Tensor:
        return torch.stack([s.ret_rel for s in self.states])

    def get_upper_wick_rel(self) -> torch.Tensor:
        return torch.stack([s.upper_wick_rel for s in self.states])

    def get_lower_wick_rel(self) -> torch.Tensor:
        return torch.stack([s.lower_wick_rel for s in self.states])

    def get_x_rel(self) -> torch.Tensor:
        return torch.stack([s.x_rel for s in self.states])

    def get_c_rel(self) -> torch.Tensor:
        return torch.stack([s.c_rel for s in self.states])

    def get_unrl_rel(self) -> torch.Tensor:
        return torch.stack([s.unrl_rel for s in self.states])

    def get_side_rel(self) -> torch.Tensor:
        return torch.stack([s.side_rel for s in self.states])

    def to_tensor(self) -> torch.Tensor:
        return torch.stack([s.to_tensor() for s in self.states])


class MultiCurrencyEnv:
    """
    Hybrid-action multi-asset long/short trading environment.

    Positions are signed: long > 0, short < 0. Shorts commit cash as collateral
    at 1:1 (no leverage) — equivalent to a linear perpetual with leverage=1.

    Action space:
        discrete action 0              -> hold
        discrete action 1 .. N         -> open LONG  asset k (auto-closes existing short)
        discrete action N+1 .. 2N      -> open SHORT asset k (auto-closes existing long)
        discrete action 2N+1 .. 3N     -> CLOSE position on asset k
        continuous action              -> fraction in [0, 1]

    Open semantics:
        commit frac * available_cash as collateral on target asset.
        Flips auto-close the full opposite position first; frac controls the new open.

    Close semantics:
        close frac * current position units on target asset.

    state_dim = 4*M*N + 9*N + 8
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
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
        eps: float = 1e-8,
    ):
        assert isinstance(N, int) and N > 0
        assert tau_p.ndim == 1 and tau_p.numel() > 0
        assert min_open_dollars >= open_fee, "min_open_dollars must be >= open_fee"
        assert reward_mode in ("return", "log"), "reward_mode must be one of ('return', 'log')"

        self.N = N
        self.dtype = dtype
        self._explicit_device = device is not None
        self.device = torch.device(device) if device is not None else tau_p.device
        self.eps = float(eps)

        self.bankruptcy_threshold = float(bankruptcy_threshold)
        self.transaction_eps = float(transaction_eps)
        self.use_dollar_volume = bool(use_dollar_volume)
        self.tax_rate = float(tax_rate)
        self.min_open_dollars = float(min_open_dollars)
        self.val_coeff = float(val_coeff)
        self.roi_coeff = float(roi_coeff)

        self.reward_mode = reward_mode
        self.done_reward_penalty = float(done_reward_penalty)

        self.o_fee = float(open_fee)
        self.c_fee = float(close_fee)

        self.tau_p = tau_p.to(device=self.device, dtype=self.dtype)
        self.M = self.tau_p.numel()

        self.save_history = bool(save_history)
        self.history = StateHistory() if self.save_history else None

        self.C0 = float(C0)

        self.state_dim = 4 * self.M * N + 9 * N + 8
        self.action_dim = 1 + 3 * self.N

        # runtime attributes
        self.C = None
        self.pos_units = None
        self.entry_price = None
        self.committed = None
        self.t = None
        self.dt = None
        self.p = None
        self.o = None
        self.v = None
        self.v_smooth = None
        self.p_smooth = None
        self.p_rel = None
        self.hl_rel = None
        self.hl_smooth = None
        self.v_rel = None
        self.vol_rel = None
        self.body_rel = None
        self.body_smooth = None
        self.ret_rel = None
        self.upper_wick_rel = None
        self.lower_wick_rel = None
        self.V_prev = None
        self.realized_cost = None
        self.realized_pnl = None

    def _state_device(self) -> torch.device:
        if isinstance(self.C, torch.Tensor):
            return self.C.device
        return self.device

    def _market_device(self, data: dict) -> torch.device:
        if isinstance(self.C, torch.Tensor):
            return self.C.device
        if self._explicit_device:
            return self.device
        for key in ("close", "high", "low", "volume"):
            value = data.get(key)
            if isinstance(value, torch.Tensor):
                return value.device
        return self.device

    # -------------------------------------------------------------------------
    # action helpers
    # -------------------------------------------------------------------------

    def encode_hold(self) -> int:
        return 0

    def encode_long(self, asset_idx: int) -> int:
        if not (0 <= asset_idx < self.N):
            raise ValueError("asset_idx out of range")
        return 1 + asset_idx

    def encode_short(self, asset_idx: int) -> int:
        if not (0 <= asset_idx < self.N):
            raise ValueError("asset_idx out of range")
        return 1 + self.N + asset_idx

    def encode_close(self, asset_idx: int) -> int:
        if not (0 <= asset_idx < self.N):
            raise ValueError("asset_idx out of range")
        return 1 + 2 * self.N + asset_idx

    def decode_action(self, action_idx: int) -> tuple[str, int | None]:
        if action_idx == 0:
            return ("hold", None)
        if 1 <= action_idx <= self.N:
            return ("long", action_idx - 1)
        if self.N + 1 <= action_idx <= 2 * self.N:
            return ("short", action_idx - 1 - self.N)
        if 2 * self.N + 1 <= action_idx < self.action_dim:
            return ("close", action_idx - 1 - 2 * self.N)
        raise ValueError(f"invalid action index {action_idx}")

    def _split_action(self, action) -> tuple[int, float]:
        if action is None:
            return 0, 0.0
        if isinstance(action, (tuple, list)):
            action_d, action_c = action
        else:
            action_d, action_c = action, 1.0

        action_idx = int(action_d.item() if isinstance(action_d, torch.Tensor) else action_d)
        frac = float(action_c.item() if isinstance(action_c, torch.Tensor) else action_c)
        return action_idx, min(max(frac, 0.0), 1.0)

    def valid_action_mask(self) -> torch.Tensor:
        """
        Boolean mask of shape [action_dim] for the discrete head.
        Assumes full-fraction execution for validity checks.
        """
        mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self._state_device())
        mask[0] = True

        cash = float(self.C.item())

        for k in range(self.N):
            units = float(self.pos_units[k].item())
            is_long = units > self.eps
            is_short = units < -self.eps
            is_flat = not (is_long or is_short)

            notional = abs(units) * float(self.p[k].item())
            close_ok = (not is_flat) and notional > self.c_fee

            if is_flat:
                cash_avail = cash
            elif close_ok:
                committed_k = float(self.committed[k].item())
                entry = float(self.entry_price[k].item())
                price = float(self.p[k].item())
                pnl = units * (price - entry)
                gross = committed_k + pnl
                proceeds = max(gross - self.c_fee, 0.0)
                pre_tax = proceeds - committed_k
                tax = max(pre_tax, 0.0) * self.tax_rate
                net_proceeds = proceeds - tax
                cash_avail = cash + net_proceeds
            else:
                cash_avail = cash

            can_open = cash_avail > self.o_fee and (cash_avail - self.o_fee) >= self.min_open_dollars

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

    # -------------------------------------------------------------------------
    # state feature computations
    # -------------------------------------------------------------------------

    def _compute_time_vector(self) -> torch.Tensor:
        time = datetime.fromtimestamp(self.t)
        dateiso = time.isocalendar()
        total_weeks = datetime(time.year, 12, 28).isocalendar().week
        period_day = (time.hour + (time.minute + time.microsecond / 1e6) / 60.0) / 24.0
        period_week = (dateiso.weekday - 1 + period_day) / 7.0
        period_year = (dateiso.week - 1 + period_week) / max(total_weeks, 1)
        angles = torch.tensor(
            [period_day, period_week, period_year],
            dtype=self.dtype,
            device=self._state_device(),
        )
        return torch.cat([torch.sin(2 * PI * angles), torch.cos(2 * PI * angles)])

    def _compute_value_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cash_rel, x_rel_per_asset)."""
        V = self.V.clamp(min=self.eps)
        cash_rel = self.C / V
        x_assets = (self.pos_units * self.p) / V
        return cash_rel, x_assets

    def _compute_cost_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        S = self.committed.sum()
        c_rel = (
            self.committed / (S + self.eps)
            if S.item() > 0
            else torch.zeros(self.N, dtype=self.dtype, device=self._state_device())
        )
        rho = S / (S + self.C + self.eps)
        return c_rel, rho

    def _compute_unrl_rel(self) -> torch.Tensor:
        has = self.committed > self.eps
        unrl_rel = torch.zeros(self.N, dtype=self.dtype, device=self._state_device())
        if has.any():
            pnl = self.pos_units * (self.p - self.entry_price)
            tax = pnl.clamp(min=0.0) * self.tax_rate
            net = pnl - tax
            unrl_rel[has] = net[has] / self.committed[has]
        return unrl_rel

    def _compute_side_rel(self) -> torch.Tensor:
        side = torch.zeros(self.N, dtype=self.dtype, device=self._state_device())
        side[self.pos_units > self.eps] = 1.0
        side[self.pos_units < -self.eps] = -1.0
        return side

    def _compute_hl_rel(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        return 2.0 * (high - low) / (high + low + self.eps)

    def _compute_bar_features(
        self,
        open_: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        close: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (body_rel, ret_rel, upper_wick_rel, lower_wick_rel) per asset."""
        rng = (high - low).clamp(min=self.eps)
        body_rel = ((close - open_) / rng).clamp(-1.0, 1.0)
        ret_rel = (close - open_) / open_.clamp(min=self.eps)
        body_high = torch.maximum(open_, close)
        body_low = torch.minimum(open_, close)
        upper_wick_rel = ((high - body_high) / rng).clamp(min=0.0)
        lower_wick_rel = ((body_low - low) / rng).clamp(min=0.0)
        return body_rel, ret_rel, upper_wick_rel, lower_wick_rel

    def _coerce_time_scalar(self, value, *, name: str = "time") -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be a scalar tensor, got shape {tuple(value.shape)}.")
            return float(value.item())
        return float(value)

    def _coerce_market_tensor(self, name: str, value, device: torch.device) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
        value = value.to(device=device, dtype=self.dtype).reshape(-1)
        if value.numel() != self.N:
            raise ValueError(f"{name} must have {self.N} elements, got {value.numel()}.")
        return value

    def _get_market_inputs(
        self,
        close=None,
        high=None,
        low=None,
        volume=None,
        time=None,
        open_=None,
        *,
        data: dict | None = None,
    ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if data is None and isinstance(close, dict):
            data = close
            close = high = low = volume = time = open_ = None

        if data is not None:
            if any(value is not None for value in (close, high, low, volume, time, open_)):
                raise ValueError("Pass either 'data' or raw market tensors, not both.")
            for key in ("close", "high", "low", "volume", "time"):
                if key not in data:
                    raise KeyError(f"data must contain '{key}'")
            device = self._market_device(data)
            close = self._coerce_market_tensor("data['close']", data["close"], device)
            high = self._coerce_market_tensor("data['high']", data["high"], device)
            low = self._coerce_market_tensor("data['low']", data["low"], device)
            volume = self._coerce_market_tensor("data['volume']", data["volume"], device)
            open_ = self._coerce_market_tensor("data['open']", data.get("open", data["close"]), device)
            t = self._coerce_time_scalar(data["time"], name="data['time']")
            return t, close, high, low, volume, open_

        missing = [
            name for name, value in (
                ("close", close),
                ("high", high),
                ("low", low),
                ("volume", volume),
                ("time", time),
            )
            if value is None
        ]
        if missing:
            raise KeyError(f"Missing market inputs: {', '.join(missing)}")
        if open_ is None:
            open_ = close

        device = self._state_device()
        if not self._explicit_device:
            for value in (close, high, low, volume, open_):
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break

        close = self._coerce_market_tensor("close", close, device)
        high = self._coerce_market_tensor("high", high, device)
        low = self._coerce_market_tensor("low", low, device)
        volume = self._coerce_market_tensor("volume", volume, device)
        open_ = self._coerce_market_tensor("open", open_, device)
        t = self._coerce_time_scalar(time)
        return t, close, high, low, volume, open_

    def _get_state(self) -> State:
        t_vec = self._compute_time_vector().to(self.dtype)
        cash_rel, x_rel = self._compute_value_weights()
        cash_rel = cash_rel.to(self.dtype)
        x_rel = x_rel.to(self.dtype)
        c_rel, rho = self._compute_cost_weights()
        unrl_rel = self._compute_unrl_rel()
        side_rel = self._compute_side_rel()
        return State(
            time=t_vec,
            cash_rel=cash_rel,
            rho=rho,
            p_rel=self.p_rel.clone(),
            v_rel=self.v_rel.clone(),
            vol_rel=self.vol_rel.clone(),
            body_smooth=self.body_smooth.clone(),
            hl_rel=self.hl_rel.clone(),
            body_rel=self.body_rel.clone(),
            ret_rel=self.ret_rel.clone(),
            upper_wick_rel=self.upper_wick_rel.clone(),
            lower_wick_rel=self.lower_wick_rel.clone(),
            x_rel=x_rel,
            c_rel=c_rel,
            unrl_rel=unrl_rel,
            side_rel=side_rel,
        )

    # -------------------------------------------------------------------------
    # portfolio property
    # -------------------------------------------------------------------------

    @property
    def V(self) -> torch.Tensor:
        pnl = self.pos_units * (self.p - self.entry_price)
        return self.C + self.committed.sum() + pnl.sum()

    # -------------------------------------------------------------------------
    # reset / update
    # -------------------------------------------------------------------------

    def reset(self, data_or_close=None, high=None, low=None, volume=None, time=None, C0: Optional[float] = None, open_=None, *, data: dict | None = None) -> State:
        if C0 is not None:
            self.C0 = float(C0)

        t, close, high, low, volume, open_ = self._get_market_inputs(
            close=data_or_close,
            high=high,
            low=low,
            volume=volume,
            time=time,
            open_=open_,
            data=data,
        )
        device = close.device

        self.C = torch.tensor(self.C0, dtype=self.dtype, device=device)
        self.pos_units   = torch.zeros(self.N, dtype=self.dtype, device=device)
        self.entry_price = torch.zeros(self.N, dtype=self.dtype, device=device)
        self.committed   = torch.zeros(self.N, dtype=self.dtype, device=device)
        self.t = t
        self.dt = 0.0
        self.p = close.clone()
        self.o = open_.clone()
        self.v = volume.clone()
        if self.use_dollar_volume:
            self.v *= self.p

        self.v_smooth = self.v[None, :].repeat(self.M, 1).clone()
        self.p_smooth = self.p[None, :].repeat(self.M, 1).clone()
        self.p_rel = torch.zeros((self.M, self.N), dtype=self.dtype, device=device)
        self.hl_rel = self._compute_hl_rel(high, low)
        self.hl_smooth = self.hl_rel[None, :].repeat(self.M, 1).clone()
        self.v_rel = torch.zeros((self.M, self.N), dtype=self.dtype, device=device)
        self.vol_rel = torch.zeros((self.M, self.N), dtype=self.dtype, device=device)

        body_rel, ret_rel, upper_wick_rel, lower_wick_rel = self._compute_bar_features(
            open_, high, low, close,
        )
        self.body_rel = body_rel
        self.ret_rel = ret_rel
        self.upper_wick_rel = upper_wick_rel
        self.lower_wick_rel = lower_wick_rel
        self.body_smooth = self.body_rel[None, :].repeat(self.M, 1).clone()

        self.V_prev = self.V.detach().clone()
        self.realized_cost = torch.zeros(self.N, dtype=self.dtype, device=device)
        self.realized_pnl  = torch.zeros(self.N, dtype=self.dtype, device=device)

        state = self._get_state()
        if self.save_history:
            self.history = StateHistory()
            self.history.append(state)
        return state

    def _update(self, close_or_data=None, high=None, low=None, volume=None, time=None, open_=None, *, data: dict | None = None) -> None:
        self.V_prev = self.V.detach().clone()
        t_new, close, high, low, volume, open_ = self._get_market_inputs(
            close=close_or_data,
            high=high,
            low=low,
            volume=volume,
            time=time,
            open_=open_,
            data=data,
        )
        if t_new < self.t:
            print("WARNING: dt < 0 - setting to zero!")
        self.dt = (t_new - self.t) * float(t_new > self.t)
        self.t = t_new
        self.p[:] = close
        self.o[:] = open_

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

        body_rel, ret_rel, upper_wick_rel, lower_wick_rel = self._compute_bar_features(
            open_, high, low, close,
        )
        self.body_rel = body_rel
        self.ret_rel = ret_rel
        self.upper_wick_rel = upper_wick_rel
        self.lower_wick_rel = lower_wick_rel
        self.body_smooth += alpha_p * (self.body_rel[None, :] - self.body_smooth)

    # -------------------------------------------------------------------------
    # trading primitives
    # -------------------------------------------------------------------------

    def _close_position(self, k: int, frac: float = 1.0) -> bool:
        """Close frac of position on asset k. Returns True if executed."""
        units = self.pos_units[k]
        if abs(float(units.item())) <= self.eps:
            return False

        units_to_close = frac * units          # signed
        price = self.p[k]
        notional = torch.abs(units_to_close) * price
        if float(notional.item()) <= self.c_fee:
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
        self.pos_units[k]  -= units_to_close
        self.committed[k]  -= realized_committed
        # entry_price stays unchanged

        self.realized_cost[k] += realized_committed
        self.realized_pnl[k]  += realized_pnl
        return True

    def _open_position(self, k: int, side: int, frac: float = 1.0) -> bool:
        """Open a side=+1 long or side=-1 short on asset k, committing frac of cash."""
        cash = float(self.C.item())
        budget = frac * cash
        buy_dollars = budget - self.o_fee
        if not (budget > self.o_fee and buy_dollars >= self.min_open_dollars):
            return False

        price_k = self.p[k].clamp(min=self.eps)
        buy_dollars_t = torch.tensor(buy_dollars, dtype=self.dtype, device=self.C.device)
        open_fee_t = torch.tensor(self.o_fee, dtype=self.dtype, device=self.C.device)

        units_mag = buy_dollars_t / price_k
        total_cost = torch.minimum(buy_dollars_t + open_fee_t, self.C)

        self.C -= total_cost
        self.pos_units[k]   += float(side) * units_mag
        self.committed[k]   += buy_dollars_t
        # VWAP entry: weighted average if adding to existing same-side position
        old_units = abs(float(self.pos_units[k].item())) - float(units_mag.item())
        if old_units > self.eps:
            old_entry = float(self.entry_price[k].item())
            new_entry = (old_units * old_entry + float(units_mag.item()) * float(price_k.item())) / abs(float(self.pos_units[k].item()))
            self.entry_price[k] = torch.tensor(new_entry, dtype=self.dtype, device=self.C.device)
        else:
            self.entry_price[k] = self.p[k].clone()
        return True

    def _trade(self, a) -> tuple[torch.Tensor, bool]:
        self.realized_cost.zero_()
        self.realized_pnl.zero_()

        action_idx, frac = self._split_action(a)

        if not (0 <= action_idx < self.action_dim):
            raise ValueError(f"Invalid discrete action {action_idx}, expected in [0, {self.action_dim - 1}]")

        # action_info = [action_type, asset_idx, frac]
        action_info = torch.tensor([0.0, -1.0, 0.0], dtype=self.dtype, device=self._state_device())

        if action_idx == 0:
            return action_info, True

        kind, k = self.decode_action(action_idx)
        action_info[1] = float(k)
        action_info[2] = frac

        if kind == "close":
            action_info[0] = 3.0
            did = self._close_position(k, frac)
            return action_info, did

        side = 1 if kind == "long" else -1
        action_info[0] = 1.0 if side == 1 else 2.0

        # flip: close full opposite position first
        units = float(self.pos_units[k].item())
        if (side == 1 and units < -self.eps) or (side == -1 and units > self.eps):
            self._close_position(k, frac=1.0)

        # reject if already same-side
        units_after = float(self.pos_units[k].item())
        if (side == 1 and units_after > self.eps) or (side == -1 and units_after < -self.eps):
            return action_info, False

        did = self._open_position(k, side, frac)

        dust = self.committed < self.transaction_eps
        self.pos_units[dust]   = 0.0
        self.committed[dust]   = 0.0
        self.entry_price[dust] = 0.0
        self.C = torch.clamp(self.C, min=0.0)

        return action_info, did

    # -------------------------------------------------------------------------
    # reward
    # -------------------------------------------------------------------------

    def _reward(self) -> float:
        if self.realized_cost.sum() > self.eps:
            reward = self.roi_coeff * (self.realized_pnl.sum() / self.realized_cost.sum()).item()
        else:
            V = self.V
            V_prev = self.V_prev.clamp(min=self.eps)
            if self.reward_mode == "return":
                reward = self.val_coeff * ((V - V_prev) / V_prev).detach().item()
            else:
                reward = self.val_coeff * torch.log(V.clamp(min=self.eps) / V_prev).detach().item()
        return float(reward)

    # -------------------------------------------------------------------------
    # main step
    # -------------------------------------------------------------------------

    def step(self, a, data_or_close=None, high=None, low=None, volume=None, time=None, open_=None, *, data: dict | None = None) -> Tuple[State, float, bool, Dict]:
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
        reward = self._reward()

        done = float(self.V) <= self.bankruptcy_threshold
        if done:
            reward -= self.done_reward_penalty

        info = {
            "t": self.t,
            "action_type": int(action_info[0].item()),   # 0=hold, 1=long, 2=short, 3=close
            "action_asset": int(action_info[1].item()),
            "action_frac": float(action_info[2].item()),
            "valid_trade": bool(valid_trade),
            "C": float(self.C),
            "V": float(self.V),
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


class BatchedMultiCurrencyEnv:
    """
    Fully-batched B-environment version of the hybrid long/short env.

    Actions: tuple of ((B,) discrete int, (B,) continuous fraction).
    Observation: (B, state_dim) tensor.

    Flattened layout matches `State.to_tensor()`:
        [ globals(8) | asset_0(4M+9) | ... | asset_{N-1}(4M+9) ]
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
        done_reward_penalty: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
        eps: float = 1e-8,
    ):
        assert isinstance(B, int) and B > 0
        assert isinstance(N, int) and N > 0
        assert tau_p.ndim == 1 and tau_p.numel() > 0
        assert min_open_dollars >= open_fee
        assert reward_mode in ("return", "log")

        self.B = B
        self.N = N
        self.dtype = dtype
        self._explicit_device = device is not None
        self.device = torch.device(device) if device is not None else tau_p.device
        self.eps = float(eps)
        self.C0 = float(C0)
        self.bankruptcy_threshold = float(bankruptcy_threshold)
        self.transaction_eps = float(transaction_eps)
        self.use_dollar_volume = bool(use_dollar_volume)
        self.tax_rate = float(tax_rate)
        self.min_open_dollars = float(min_open_dollars)
        self.val_coeff = float(val_coeff)
        self.roi_coeff = float(roi_coeff)
        self.reward_mode = reward_mode
        self.done_reward_penalty = float(done_reward_penalty)
        self.o_fee = float(open_fee)
        self.c_fee = float(close_fee)

        self.tau_p = tau_p.to(device=self.device, dtype=dtype)
        self.M = self.tau_p.numel()

        self.state_dim = 4 * self.M * N + 9 * N + 8
        self.action_dim = 1 + 3 * N

        self._b_idx = torch.arange(B, device=self.tau_p.device)

        # runtime state
        self.C = None
        self.pos_units = None
        self.entry_price = None
        self.committed = None
        self.p = None
        self.o = None
        self.v = None
        self.p_smooth = None
        self.v_smooth = None
        self.hl_rel = None
        self.hl_smooth = None
        self.p_rel = None
        self.v_rel = None
        self.vol_rel = None
        self.body_rel = None
        self.body_smooth = None
        self.ret_rel = None
        self.upper_wick_rel = None
        self.lower_wick_rel = None
        self.V_prev = None
        self.realized_cost = None
        self.realized_pnl = None
        self.t = None

    def _state_device(self) -> torch.device:
        if isinstance(self.C, torch.Tensor):
            return self.C.device
        return self.device

    def _coerce_market_tensor(self, name: str, value, device: torch.device) -> torch.Tensor:
        value = torch.as_tensor(value, device=device, dtype=self.dtype)
        expected_shape = (self.B, self.N)
        if value.shape == expected_shape:
            return value
        if value.numel() == self.B * self.N:
            return value.reshape(expected_shape)
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(value.shape)}. "
            f"Check that the environment N={self.N} matches the market data asset count."
        )

    # -------------------------------------------------------------------------
    # action helpers
    # -------------------------------------------------------------------------

    def encode_hold(self) -> int:
        return 0

    def encode_long(self, asset_idx: int) -> int:
        return 1 + asset_idx

    def encode_short(self, asset_idx: int) -> int:
        return 1 + self.N + asset_idx

    def encode_close(self, asset_idx: int) -> int:
        return 1 + 2 * self.N + asset_idx

    # -------------------------------------------------------------------------
    # portfolio property
    # -------------------------------------------------------------------------

    @property
    def V(self) -> torch.Tensor:
        pnl = self.pos_units * (self.p - self.entry_price)
        return self.C + self.committed.sum(dim=-1) + pnl.sum(dim=-1)

    # -------------------------------------------------------------------------
    # reset
    # -------------------------------------------------------------------------

    def reset(self, close, high, low, volume, time, open_=None, C0=None):
        if C0 is not None:
            self.C0 = float(C0)
        if open_ is None:
            open_ = close
        B, N, M = self.B, self.N, self.M

        if self._explicit_device:
            device = self.device
        else:
            device = close.device if isinstance(close, torch.Tensor) else self.device
        close  = self._coerce_market_tensor("close", close, device)
        high   = self._coerce_market_tensor("high", high, device)
        low    = self._coerce_market_tensor("low", low, device)
        volume = self._coerce_market_tensor("volume", volume, device)
        open_  = self._coerce_market_tensor("open_", open_, device)
        self.t = (
            time.to(device=device, dtype=torch.float64)
            if isinstance(time, torch.Tensor)
            else torch.tensor(time, dtype=torch.float64, device=device)
        )
        self._b_idx = torch.arange(B, device=device)

        self.C           = torch.full((B,), self.C0, dtype=self.dtype, device=device)
        self.pos_units   = torch.zeros(B, N, dtype=self.dtype, device=device)
        self.entry_price = torch.zeros(B, N, dtype=self.dtype, device=device)
        self.committed   = torch.zeros(B, N, dtype=self.dtype, device=device)
        self.p           = close.clone()
        self.o           = open_.clone()
        self.v           = volume.clone()
        if self.use_dollar_volume:
            self.v = self.v * self.p

        self.p_smooth  = self.p[:, None, :].expand(B, M, N).clone()
        self.v_smooth  = self.v[:, None, :].expand(B, M, N).clone()
        self.hl_rel    = self._hl_rel(high, low)
        self.hl_smooth = self.hl_rel[:, None, :].expand(B, M, N).clone()

        self.p_rel   = torch.zeros(B, M, N, dtype=self.dtype, device=device)
        self.v_rel   = torch.zeros(B, M, N, dtype=self.dtype, device=device)
        self.vol_rel = torch.zeros(B, M, N, dtype=self.dtype, device=device)

        body_rel, ret_rel, upper_wick_rel, lower_wick_rel = self._bar_features(open_, high, low, close)
        self.body_rel       = body_rel
        self.ret_rel        = ret_rel
        self.upper_wick_rel = upper_wick_rel
        self.lower_wick_rel = lower_wick_rel
        self.body_smooth    = self.body_rel[:, None, :].expand(B, M, N).clone()

        self.V_prev        = self.C.clone()
        self.realized_cost = torch.zeros(B, N, dtype=self.dtype, device=device)
        self.realized_pnl  = torch.zeros(B, N, dtype=self.dtype, device=device)
        return self._obs()

    # -------------------------------------------------------------------------
    # internal helpers
    # -------------------------------------------------------------------------

    def _hl_rel(self, high, low):
        return 2.0 * (high - low) / (high + low + self.eps)

    def _bar_features(self, open_, high, low, close):
        rng = (high - low).clamp(min=self.eps)
        body_rel = ((close - open_) / rng).clamp(-1.0, 1.0)
        ret_rel = (close - open_) / open_.clamp(min=self.eps)
        body_high = torch.maximum(open_, close)
        body_low = torch.minimum(open_, close)
        upper_wick_rel = ((high - body_high) / rng).clamp(min=0.0)
        lower_wick_rel = ((body_low - low) / rng).clamp(min=0.0)
        return body_rel, ret_rel, upper_wick_rel, lower_wick_rel

    def _time_features(self):
        features = []
        device = self._state_device()
        for ts in self.t.detach().cpu().tolist():
            dt_obj = datetime.fromtimestamp(ts)
            iso = dt_obj.isocalendar()
            total_weeks = datetime(dt_obj.year, 12, 28).isocalendar().week
            period_day  = (dt_obj.hour + (dt_obj.minute + dt_obj.microsecond / 1e6) / 60.0) / 24.0
            period_week = (iso.weekday - 1 + period_day) / 7.0
            period_year = (iso.week - 1 + period_week) / max(total_weeks, 1)
            angles = torch.tensor([period_day, period_week, period_year], dtype=self.dtype, device=device)
            features.append(torch.cat([torch.sin(2 * PI * angles), torch.cos(2 * PI * angles)]))
        return torch.stack(features)

    def _obs(self):
        B, N = self.B, self.N
        V = self.V.clamp(min=self.eps)

        t_vec    = self._time_features()                          # (B, 6)
        cash_rel = (self.C / V)                                   # (B,)
        x_assets = (self.pos_units * self.p) / V[:, None]         # (B, N)

        S     = self.committed.sum(dim=1)
        c_rel = torch.where(
            S[:, None] > 0,
            self.committed / (S[:, None] + self.eps),
            torch.zeros(B, N, dtype=self.dtype, device=self._state_device()),
        )
        rho   = S / (S + self.C + self.eps)                       # (B,)

        has      = self.committed > self.eps
        pnl      = self.pos_units * (self.p - self.entry_price)
        tax      = pnl.clamp(min=0.0) * self.tax_rate
        net      = pnl - tax
        unrl_rel = torch.where(
            has,
            net / self.committed.clamp(min=self.eps),
            torch.zeros(B, N, dtype=self.dtype, device=self._state_device()),
        )

        side_rel = torch.zeros(B, N, dtype=self.dtype, device=self._state_device())
        side_rel[self.pos_units > self.eps]  = 1.0
        side_rel[self.pos_units < -self.eps] = -1.0

        # per-asset (M, N) stack -> (B, 4M, N) -> (B, N, 4M)
        mn = torch.cat([self.p_rel, self.v_rel, self.vol_rel, self.body_smooth], dim=1).transpose(1, 2)
        # per-asset scalars -> (B, N, 9)
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
        asset_block = torch.cat([mn, scalars], dim=2)             # (B, N, 4M+9)

        globals_block = torch.cat([t_vec, cash_rel[:, None], rho[:, None]], dim=1)  # (B, 8)

        return torch.cat([globals_block, asset_block.reshape(B, -1)], dim=1)

    def _update(self, close, high, low, volume, time, open_=None):
        self.V_prev = self.V.detach().clone()
        device = self._state_device()
        if open_ is None:
            open_ = close
        close  = close.to(device=device, dtype=self.dtype)
        high   = high.to(device=device, dtype=self.dtype)
        low    = low.to(device=device, dtype=self.dtype)
        volume = volume.to(device=device, dtype=self.dtype)
        open_  = open_.to(device=device, dtype=self.dtype)
        t_new  = (
            time.to(device=device, dtype=torch.float64)
            if isinstance(time, torch.Tensor)
            else torch.tensor(time, dtype=torch.float64, device=device)
        )

        dt = (t_new - self.t).clamp(min=0.0).to(self.dtype)
        self.t = t_new
        self.p = close
        self.o = open_

        alpha_p = 1.0 - torch.exp(-dt[:, None, None] / self.tau_p[None, :, None].clamp(min=self.eps))
        p_exp = self.p[:, None, :]
        self.p_smooth = self.p_smooth + alpha_p * (p_exp - self.p_smooth)
        self.p_rel    = (p_exp - self.p_smooth) / self.p_smooth.clamp(min=self.eps)

        hl_new = self._hl_rel(high, low)
        self.hl_rel = hl_new
        hl_exp = hl_new[:, None, :]
        self.hl_smooth = self.hl_smooth + alpha_p * (hl_exp - self.hl_smooth)
        self.vol_rel   = (hl_exp - self.hl_smooth) / self.hl_smooth.clamp(min=self.eps)

        self.v = volume * self.p if self.use_dollar_volume else volume
        v_exp = self.v[:, None, :]
        self.v_smooth = self.v_smooth + alpha_p * (v_exp - self.v_smooth)
        self.v_rel    = (v_exp - self.v_smooth) / self.v_smooth.clamp(min=self.eps)

        body_rel, ret_rel, upper_wick_rel, lower_wick_rel = self._bar_features(open_, high, low, close)
        self.body_rel       = body_rel
        self.ret_rel        = ret_rel
        self.upper_wick_rel = upper_wick_rel
        self.lower_wick_rel = lower_wick_rel
        body_exp = body_rel[:, None, :]
        self.body_smooth = self.body_smooth + alpha_p * (body_exp - self.body_smooth)

    def _split_actions(self, actions):
        device = self._state_device()
        if actions is None:
            actions_d = torch.zeros(self.B, dtype=torch.long, device=device)
            actions_c = torch.zeros(self.B, dtype=self.dtype, device=device)
        elif isinstance(actions, tuple):
            actions_d, actions_c = actions
        elif isinstance(actions, list) and len(actions) == 2 and not all(isinstance(x, (int, float, bool)) for x in actions):
            actions_d, actions_c = actions
        else:
            actions_d = actions
            actions_c = torch.ones_like(torch.as_tensor(actions_d, device=device), dtype=self.dtype)

        actions_d = torch.as_tensor(actions_d).to(device=device, dtype=torch.long).reshape(-1)
        actions_c = torch.as_tensor(actions_c).to(device=device, dtype=self.dtype).reshape(-1).clamp(0.0, 1.0)

        if actions_d.numel() != self.B:
            raise ValueError(f"actions_d must have {self.B} elements, got {actions_d.numel()}")
        if actions_c.numel() != self.B:
            raise ValueError(f"actions_c must have {self.B} elements, got {actions_c.numel()}")
        if bool(((actions_d < 0) | (actions_d >= self.action_dim)).any().item()):
            raise ValueError(f"Invalid discrete action, expected values in [0, {self.action_dim - 1}]")
        return actions_d, actions_c

    def _trade(self, actions) -> None:
        """
        Two-phase vectorized execution: close (incl. flip-triggered) then open.
        Fraction from continuous head scales open budget and close size.
        """
        B, N = self.B, self.N
        b = self._b_idx
        actions_d, frac = self._split_actions(actions)

        self.realized_cost = torch.zeros(B, N, dtype=self.dtype, device=self._state_device())
        self.realized_pnl  = torch.zeros(B, N, dtype=self.dtype, device=self._state_device())

        sell_offset  = 1 + N
        close_offset = 1 + 2 * N

        is_long_act  = (actions_d >= 1)            & (actions_d < sell_offset)
        is_short_act = (actions_d >= sell_offset)  & (actions_d < close_offset)
        is_close_act = (actions_d >= close_offset) & (actions_d < self.action_dim)

        long_asset  = (actions_d - 1).clamp(0, N - 1)
        short_asset = (actions_d - sell_offset).clamp(0, N - 1)
        close_asset = (actions_d - close_offset).clamp(0, N - 1)

        cur_short_on_long = self.pos_units[b, long_asset]  < -self.eps
        cur_long_on_short = self.pos_units[b, short_asset] > self.eps

        flip_long  = is_long_act  & cur_short_on_long
        flip_short = is_short_act & cur_long_on_short

        do_close = is_close_act | flip_long | flip_short

        target_close_asset = torch.where(
            is_close_act, close_asset,
            torch.where(flip_long, long_asset, short_asset),
        )

        # Close fraction: for explicit close use continuous frac; for flips close fully
        close_frac = torch.where(
            is_close_act,
            frac,
            torch.ones(B, dtype=self.dtype, device=self._state_device()),
        )

        units_c     = self.pos_units[b, target_close_asset]
        entry_c     = self.entry_price[b, target_close_asset]
        committed_c = self.committed[b, target_close_asset]
        price_c     = self.p[b, target_close_asset]

        units_to_close = close_frac * units_c                     # signed
        notional_c = torch.abs(units_to_close) * price_c
        pos_frac_c = (torch.abs(units_to_close) / torch.abs(units_c).clamp(min=self.eps)).clamp(0.0, 1.0)
        realized_committed = pos_frac_c * committed_c

        close_valid = do_close & (torch.abs(units_c) > self.eps) & (notional_c > self.c_fee)

        pnl_c       = units_to_close * (price_c - entry_c)
        gross_c     = realized_committed + pnl_c
        proceeds    = (gross_c - self.c_fee).clamp(min=0.0)
        pre_tax_pnl = proceeds - realized_committed
        tax         = pre_tax_pnl.clamp(min=0.0) * self.tax_rate
        net_proceeds = proceeds - tax
        realized_pnl = net_proceeds - realized_committed

        cv = close_valid.to(self.dtype)
        self.C = self.C + net_proceeds * cv

        self.pos_units.scatter_add_(1, target_close_asset[:, None], -(units_to_close * cv)[:, None])
        self.committed.scatter_add_(1, target_close_asset[:, None], -(realized_committed * cv)[:, None])
        # entry_price unchanged for partial close

        self.realized_cost.scatter_add_(1, target_close_asset[:, None], (realized_committed * cv)[:, None])
        self.realized_pnl.scatter_add_(1, target_close_asset[:, None], (realized_pnl * cv)[:, None])

        # Zero entry_price for fully-closed slots so scatter_add_ in open phase
        # starts from 0 (prevents stacking new price on stale entry).
        flat_after_close = torch.abs(self.pos_units) <= self.eps
        self.entry_price = self.entry_price.masked_fill(flat_after_close, 0.0)

        flip_failed_long  = flip_long  & ~close_valid
        flip_failed_short = flip_short & ~close_valid

        # ---- OPEN phase ----
        is_open = (is_long_act & ~flip_failed_long) | (is_short_act & ~flip_failed_short)
        open_asset = torch.where(is_long_act, long_asset, short_asset)
        open_side  = torch.where(
            is_long_act,
            torch.ones(B, dtype=self.dtype, device=self._state_device()),
            -torch.ones(B, dtype=self.dtype, device=self._state_device()),
        )

        cur_units_open = self.pos_units[b, open_asset]
        slot_flat = torch.abs(cur_units_open) <= self.eps

        buy_budget  = frac * self.C                                 # fractional commitment
        buy_dollars = buy_budget - self.o_fee
        open_valid  = is_open & slot_flat & (buy_budget > self.o_fee) & (buy_dollars >= self.min_open_dollars)

        open_price = self.p[b, open_asset].clamp(min=self.eps)
        units_mag  = buy_dollars / open_price
        open_units = open_side * units_mag
        total_cost = torch.minimum(buy_budget, self.C)

        ov = open_valid.to(self.dtype)
        self.C = self.C - total_cost * ov
        self.pos_units.scatter_add_(1, open_asset[:, None], (open_units * ov)[:, None])
        self.committed.scatter_add_(1, open_asset[:, None], (buy_dollars * ov)[:, None])
        self.entry_price.scatter_add_(1, open_asset[:, None], (open_price * ov)[:, None])

        # Dust cleanup
        dust = self.committed < self.transaction_eps
        self.pos_units   = self.pos_units.masked_fill(dust, 0.0)
        self.committed   = self.committed.masked_fill(dust, 0.0)
        self.entry_price = self.entry_price.masked_fill(dust, 0.0)
        self.C           = self.C.clamp(min=0.0)

    def _reward(self):
        has_realized = self.realized_cost.sum(dim=1) > self.eps
        roi = self.realized_pnl.sum(dim=1) / self.realized_cost.sum(dim=1).clamp(min=self.eps)
        V_curr = self.V
        V_prev = self.V_prev.clamp(min=self.eps)
        if self.reward_mode == "return":
            baseline = (V_curr - V_prev) / V_prev
        else:
            baseline = torch.log(V_curr.clamp(min=self.eps) / V_prev)
        return torch.where(has_realized, self.roi_coeff * roi, self.val_coeff * baseline)

    # -------------------------------------------------------------------------
    # public step / mask
    # -------------------------------------------------------------------------

    def step(self, actions, close, high, low, volume, time, open_=None):
        """Returns obs (B, state_dim), rewards (B,), dones (B,)."""
        self._trade(actions)
        self._update(close, high, low, volume, time, open_)
        rewards = self._reward()
        dones   = self.V <= self.bankruptcy_threshold
        rewards = torch.where(dones, rewards - self.done_reward_penalty, rewards)
        return self._obs(), rewards, dones

    def valid_action_mask(self):
        """Returns (B, action_dim) bool tensor."""
        B, N = self.B, self.N
        eps = self.eps

        mask = torch.zeros(B, self.action_dim, dtype=torch.bool, device=self._state_device())
        mask[:, 0] = True

        is_long  = self.pos_units > eps
        is_short = self.pos_units < -eps
        is_flat  = ~(is_long | is_short)

        notional = torch.abs(self.pos_units) * self.p
        close_ok = (~is_flat) & (notional > self.c_fee)

        pnl      = self.pos_units * (self.p - self.entry_price)
        gross    = self.committed + pnl
        proceeds = (gross - self.c_fee).clamp(min=0.0)
        pre_tax  = proceeds - self.committed
        tax      = pre_tax.clamp(min=0.0) * self.tax_rate
        net_proc = proceeds - tax
        cash_if_close = torch.where(close_ok, self.C[:, None] + net_proc, self.C[:, None].expand(B, N))
        cash_avail = torch.where(is_flat, self.C[:, None].expand(B, N), cash_if_close)

        can_open = (cash_avail > self.o_fee) & ((cash_avail - self.o_fee) >= self.min_open_dollars)

        long_valid  = (~is_long)  & can_open & (is_flat | (is_short & close_ok))
        short_valid = (~is_short) & can_open & (is_flat | (is_long  & close_ok))

        mask[:, 1:1 + N]             = long_valid
        mask[:, 1 + N:1 + 2 * N]     = short_valid
        mask[:, 1 + 2 * N:1 + 3 * N] = close_ok
        return mask


__all__ = [
    "State",
    "StateHistory",
    "MultiCurrencyEnv",
    "BatchedMultiCurrencyEnv",
]
