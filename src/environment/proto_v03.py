from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch

PI = 3.141592653589793238462


@dataclass
class State:
    """
    Observation returned by the environment.

    Attributes
    ----------
    time : torch.Tensor
        Cyclic time features, shape [3,2].
    p_rel : torch.Tensor
        Relative price deviations across time-constants, shape [M_p, N],
        p_rel[m, i] = (p[i] - s[m, i]) / s[m, i].
    x_rel : torch.Tensor
        Value weights, shape [N+1]: [ cash_frac, (w * p) / V ].
    s_rel : Optional[torch.Tensor]
        EWMA volatility, shape [M_v, N], or None if disabled.
    v_rel : Optional[torch.Tensor]
        Relative (dollar) v vs EWMA baseline, shape [M_u, N], or None.
    """
    time: torch.Tensor
    p_rel: torch.Tensor
    x_rel: torch.Tensor
    v_rel: torch.Tensor

    def to_tensor(self) -> torch.Tensor:
        """Concatenate into a 1-D tensor; matches env.state_size."""
        parts = [
            self.time.flatten(),
            self.p_rel.flatten(),
            self.x_rel.flatten(),
            self.v_rel.flatten(),
            self.s_rel.flatten(),
        ]
        return torch.cat(parts)


class StateHistory:
    """Simple container for storing a history of `State`s."""

    def __init__(self):
        self.states: list[State] = []

    def append(self, state: State):
        self.states.append(state)

    def get_time(self) -> torch.Tensor:
        return torch.stack([s.time for s in self.states])

    def get_p_rel(self) -> torch.Tensor:
        return torch.stack([s.p_rel for s in self.states])

    def get_x_rel(self) -> torch.Tensor:
        return torch.stack([s.x_rel for s in self.states])

    def get_v(self) -> torch.Tensor:
        return torch.stack([s.v_rel for s in self.states])

    def to_tensor(self) -> torch.Tensor:
        return torch.stack([s.to_tensor() for s in self.states])


class MultiCurrencyEnv:
    r"""
    Gym-like multi-currency trading environment (no gym dependency).

    Observation (state)
    -------------------
      • time   : [3,2] cyclic time features (day/week/year).
      • p_rel  : [M_p, N]  relative price deviations (p - s)/s with multi-τ price EWMAs.
      • x      : [N+1]     value weights: [C/V, (w * p)/V].
      • s_rel  : [M_v, N]  (optional) EWMA volatility √EWMA(r²).
      • v_rel  : [M_u, N]  (optional) relative (dollar) v vs EWMA baseline.

    Actions
    -------
    a ∈ (-1,1)^{N+1}, where:
      • a[0]  = buy budget fraction (mapped to [0,1)).
      • a[1:] = per-asset signals: negative => sell fraction of *units*; positive => share of buy budget.

    Reward modes
    ------------
      • "diff" : V_{t+1} − V_t         (USD PnL).
      • "log"  : log((V_{t+1}+eps)/(V_t+eps)).
      • "realized_roi" : only when sells occur; sum over assets of
            ROI_i = realized_PnL_i / (realized_cost_i + eps).
        (Average-cost basis; includes buy & sell fees.)
    """

    def __init__(
        self,
        N: int,
        C0: float,
        sell_fee: float | torch.Tensor,
        buy_fee: float | torch.Tensor,
        tau_p: torch.Tensor,
        *,
        # optional
        bankruptcy_threshold: float = 0.0,
        transaction_eps: float = 0.0,
        reward_mode: str = "log",
        roi_clip: tuple[float, float] = (-1.0, 1.0),
        use_dollar_volume: bool = True,
        save_history: bool = False,
        # numerics
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-12,
    ):
        # shape checks
        assert isinstance(N, int) and N > 0
        assert tau_p.ndim == 1 and tau_p.numel() > 0
        assert reward_mode in ("diff", "log", "realized_roi")

        # basic properties
        self.N = N
        self.dtype = dtype
        self.eps = float(eps)

        # controls
        self.bankruptcy_threshold = float(bankruptcy_threshold)
        self.reward_mode = reward_mode
        self.roi_clip = tuple(roi_clip)
        self.use_dollar_volume = bool(use_dollar_volume)
        self.transaction_eps = float(transaction_eps)

        # fees (scalar → broadcast)
        self.s_fee = self._as_fee_tensor(sell_fee) # [N]
        self.b_fee = self._as_fee_tensor(buy_fee)  # [N]

        # τ-grids
        self.tau_p = tau_p.to(self.dtype)
        self.M = self.tau_p.numel()

        # history
        self.save_history = save_history
        self.history = StateHistory() if save_history else None

        # initial capital (wallets start at zero)
        self.C0 = float(C0)

        self.state_size = 7 + self.M * N + 2 * N
        self.action_size = N + 1

    def _as_fee_tensor(self, fee: float | torch.Tensor) -> torch.Tensor:
        """
        Accept a scalar 'static' fee (same for all assets) or a per-asset tensor.
        Returns a length-N tensor clamped to [0,1].
        """
        if isinstance(fee, torch.Tensor):
            assert fee.numel() in (1, self.N)
            if fee.numel() == 1:
                fee = fee.expand(self.N)
            return fee.to(self.dtype).clamp(0.0, 1.0)
        else:
            return torch.full((self.N,), float(fee), dtype=self.dtype).clamp(0.0, 1.0)
        
    def _compute_time_vector(self) -> torch.Tensor:
        """
        Map a Unix timestamp to cyclic clock coordinates for day/week/year.
        """
        time = datetime.fromtimestamp(self.t)
        dateiso = time.isocalendar()

        # ISO week count for the year (guard against edge cases)
        total_weeks = datetime(time.year, 12, 28).isocalendar().week

        period_day = (time.hour + (time.minute + time.microsecond / 1e6) / 60.0) / 24.0
        period_week = (dateiso.weekday - 1 + period_day) / 7.0
        period_year = (dateiso.week - 1 + period_week) / max(total_weeks, 1)

        angles = torch.tensor([period_day, period_week, period_year], dtype=torch.float32)
        
        return torch.concat([torch.sin(2 * PI * angles), torch.cos(2 * PI * angles)])

    def _compute_value_weights(self) -> torch.Tensor:
        """
        x = [C/V, (w*p)/V] ∈ R^{N+1}.
        """
        # get total portfolio value 
        V = self.V.clamp(min=self.eps)
        
        # get cash fraction
        x_cash = (self.C / V)[None]

        # get fraction of each asset
        x_assets = (self.w * self.p) / V
        
        return torch.cat([x_cash, x_assets])

    def _get_state(self) -> State:
        """
        Assemble current observation from internals.
        """
        t_vec = self._compute_time_vector().to(self.dtype)
        x_rel = self._compute_value_weights().to(self.dtype)
        
        return State(
            time  = t_vec,              # time-vector
            x_rel = x_rel,              # relative value of cash and assets
            p_rel = self.p_rel.clone(), # relative price change
            v_rel = self.v_rel.clone(), # log relative volume
            s_rel = self.s_rel.clone()
        )

    @property
    def V(self) -> torch.Tensor:
        return self.C + torch.sum(self.w * self.p)

    def reset(self, data: dict, C0: Optional[float] = None) -> State:
        """
        Reset environment with an initial market snapshot.

        Parameters
        ----------
        data : dict
            Must contain:
              - 'time'   : float timestamp (Unix seconds)
              - 'prices' : torch.Tensor [N]
              - 'volume' : torch.Tensor [N]
        C0 : Optional[float]
            Override initial capital (USD).

        Returns
        -------
        State
            Initial observation (p_rel≈0, x from current holdings, s_rel/v_rel initialized).
        """
        if C0 is not None:
            self.C0 = float(C0)
        self.C = torch.tensor(self.C0, dtype=self.dtype)
        self.w = torch.zeros(self.N, dtype=self.dtype)

        # ingest snapshot
        self.t = float(data["time"])
        self.p = data["prices"].to(self.dtype)
        self.v = data["volume"].to(self.dtype)
        self.v_prev = self.v.clone()

        # price trackers
        self.p_smooth = self.p[None, :].repeat(self.M, 1).clone()
        self.p_rel = torch.zeros((self.M, self.N), dtype=self.dtype)

        # volume tracker
        self.v_rel = torch.zeros((self.N,), dtype=self.dtype)

        # volatility tracker
        self.s_rel = torch.zeros((self.N,), dtype=self.dtype)

        # reward bookkeeping
        self.V_prev = self.V.detach().clone()

        # average-cost-basis bookkeeping for realized PnL
        self.invested = torch.zeros(self.N, dtype=self.dtype)  # dollars spent incl. buy fees

        state = self._get_state()
        if self.save_history:
            self.history = StateHistory()
            self.history.append(state)
        
        return state
    
    def _update(self, data: dict) -> None:
        """
        Update smoothers/features from *current* self.p and self.v.
        Assumes both price and volume snapshots have been ingested (volume is ignored if tau_v is None).
        """
        # ingest time data
        t_new = float(data["time"])
        if t_new < self.t:
            print("WARNING: dt < 0 - setting to zero!")
        self.dt = (t_new - self.t) * float(t_new > self.t)
        self.t = t_new

        # ingest price data
        self.p[:] = data["prices"].to(self.dtype)
        alpha_p = 1 - torch.exp(-self.dt / self.tau_p)[:, None]
        self.p_smooth += alpha_p * (self.p[None, :] - self.p_smooth)
        self.p_rel = (self.p[None, :] - self.p_smooth) / self.p_smooth.clamp(min=self.eps)

        # ingest volume data
        self.v_prev[:] = self.v.clone()
        self.v[:] = data["volume"].to(self.dtype)
        if self.use_dollar_volume:
            self.v[:] *= self.p
        self.v_rel[:] = torch.log(self.v / self.v_prev)

    def _trade(self, a: torch.Tensor) -> None:
        assert a.shape[-1] == self.N + 1, "Action must have N+1 outputs"

        # unpack trade fractions
        a0, a_cur = a[0], a[1:]

        # selling
        sell_mask = a_cur < -self.transaction_eps
        f_sell = torch.zeros_like(a_cur)
        f_sell[sell_mask] = (-a_cur[sell_mask]).clamp(0.0, 1.0)

        self.realized_pnl = torch.zeros(self.N, dtype=self.dtype)
        self.realized_cost = torch.zeros(self.N, dtype=self.dtype)

        if torch.any(sell_mask):
            units_sold = f_sell * self.w
            notional = units_sold * self.p
            fees_s = self.s_fee * notional
            proceeds = notional - fees_s

            # realized PnL via average-cost basis
            has_position = self.w > self.eps
            avg_cost = torch.zeros_like(self.invested)
            avg_cost[has_position] = self.invested[has_position] / self.w[has_position]
            base = units_sold * avg_cost  # dollars of cost closed
            pnl = proceeds - base

            # update books
            self.C = self.C + proceeds.sum()
            self.w = self.w - units_sold
            self.invested = (self.invested - base).clamp(min=0.0)
            dust_mask = self.w <= self.eps
            if torch.any(dust_mask):
                self.w[dust_mask] = 0.0
                self.invested[dust_mask] = 0.0

            self.realized_pnl[sell_mask] = pnl[sell_mask]
            self.realized_cost[sell_mask] = base[sell_mask]

        # buying
        buy_mask = a_cur > self.transaction_eps
        if torch.any(buy_mask):
            positives = a_cur[buy_mask]
            denom = positives.sum()
            if denom.item() > 1e-12:
                weights = torch.zeros_like(a_cur)
                weights[buy_mask] = positives / denom
                a0_bar = 0.5 * (a0 + 1.0)
                a0_bar = a0_bar.clamp(0.0, 1.0)
                I = a0_bar * self.C
                if I.item() > 0.0:
                    A = weights * I
                    fee_b = self.b_fee * A
                    # effective units bought at execution price
                    units_bought = (A - fee_b) / self.p.clamp(min=1e-12)
                    self.w = self.w + units_bought
                    self.C = self.C - I
                    # average-cost basis (invested dollars include buy fees)
                    self.invested = self.invested + A

    def _reward(self):
        r'''Calculate the reward signal.'''
        reward = None

        if self.reward_mode == "diff":
            reward = float((self.V - self.V_prev).detach())
            self.V_prev = self.V.detach()
        
        elif self.reward_mode == "log":
            V_prev = float(self.V_prev.detach())
            V_now = float(self.V.detach())
            reward = float(torch.log(torch.tensor((V_now + self.eps) / (V_prev + self.eps))).item())
            self.V_prev = self.V.detach()
        
        elif self.reward_mode == "realized_roi":
            B = self.realized_cost.sum().item()
            if B <= 0.0:
                reward = 0.0
            else:
                roi = (self.realized_pnl / (self.realized_cost + self.eps)).clamp(self.roi_clip[0], self.roi_clip[1])
                reward = float(roi.sum().detach())  # sum over assets (simple, scale-free)
        
        else:
            raise NotImplementedError(f"The reward mode {self.reward_mode} is not implemented.")
        
        return reward

    def step(self, a: torch.Tensor, data: dict) -> Tuple[State, float, bool, Dict]:
        """
        Execute trades at previous prices, then advance to the new snapshot.

        Parameters
        ----------
        a : torch.Tensor [N+1]
            Actions in (-1,1)^{N+1}.
        data : dict
            Next snapshot: {'time': float, 'prices': Tensor[N], 'volume': Tensor[N]}

        Returns
        -------
        state, reward, done, info
        """
        # trade using action a
        self._trade(a)

        # advance to new state using data
        self._update(data)

        # calculate reward signal
        reward = self._reward()

        # termination & bookkeeping
        done = float(self.V) <= self.bankruptcy_threshold
        info = {
            "t": self.t,
            "a": a.clone(),
            "C": float(self.C.clone()),
            "V": float(self.V.clone()),
            "w": self.w.clone(),
            "p": self.p.clone()
        }

        # fetch new state and store in hos
        next_state = self._get_state()
        if self.save_history and self.history is not None:
            self.history.append(next_state)

        return next_state, reward, done, info
