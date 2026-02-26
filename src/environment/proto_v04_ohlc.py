from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch

PI = 3.141592653589793238462

@dataclass
class State:
    time: torch.Tensor            # [3,2] flattened or keep [3,2]
    p_rel: torch.Tensor           # [M, N]
    x_rel: torch.Tensor           # [N+1]  (exposure: cash + value weights)
    c_rel: torch.Tensor           # [N]    (invested weights, stable)
    rho:  torch.Tensor            # []     (overall commitment)
    mny:  torch.Tensor            # [N]    (moneyness vs. entry)
    v_rel: torch.Tensor           # [N] or [M_u,N] per your design

    def to_tensor(self):
        return torch.cat([
            self.time.flatten(), self.p_rel.flatten(),
            self.x_rel.flatten(), self.c_rel.flatten(),
            self.rho.view(1), self.mny.flatten(),
            self.v_rel.flatten()
        ])


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
        temp: float = 1.0,
        bankruptcy_threshold: float = 0.0,
        transaction_eps: float = 0.0,
        reward_mode: str = "log",
        use_dollar_volume: bool = True,
        save_history: bool = False,
        # numerics
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-12,
    ):
        # shape checks
        assert isinstance(N, int) and N > 0
        assert tau_p.ndim == 1 and tau_p.numel() > 0
        assert reward_mode in ("diff", "log", "realized_roi", "smooth_roi")

        # basic properties
        self.N = N
        self.dtype = dtype
        self.eps = float(eps)

        # controls
        self.temp = float(temp)
        self.bankruptcy_threshold = float(bankruptcy_threshold)
        self.reward_mode = reward_mode
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

        self.state_size = self.M*N + 4*N + 8
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
    
    def _compute_cost_weights(self):
        S = self.invested.sum()
        c_rel = (self.invested / (S + self.eps)) if S > 0 else torch.zeros(self.N, dtype=self.dtype)
        # rho = S / (self.C0 + self.eps)
        rho = S / (S + self.C)
        return c_rel, rho
    
    def _compute_moneyness(self):
        has_pos = self.w > self.eps
        avg_cost = torch.zeros_like(self.invested)
        avg_cost[has_pos] = self.invested[has_pos] / self.w[has_pos]
        m = torch.zeros(self.N, dtype=self.dtype)
        m[has_pos] = (self.p[has_pos] - avg_cost[has_pos]) / (avg_cost[has_pos] + self.eps)
        return m

    def _get_state(self) -> State:
        t_vec = self._compute_time_vector().to(self.dtype)
        x_rel = self._compute_value_weights().to(self.dtype)      # exposure
        c_rel, rho = self._compute_cost_weights()                 # commitment (stable)
        mny = self._compute_moneyness()                           # PnL vs entry
        return State(
            time=t_vec,
            p_rel=self.p_rel.clone(),
            v_rel=self.v_rel.clone(),
            x_rel=x_rel,
            c_rel=c_rel,
            rho=rho,
            mny=mny,
        )

    @property
    def V(self) -> float:
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

        # volume trackers
        self.v_rel = torch.zeros((self.N,), dtype=self.dtype)

        # reward bookkeeping
        self.V_prev = self.V.detach().clone()
        self.V_smooth = self.V.repeat(self.M).clone()

        # average-cost-basis bookkeeping for realized PnL
        self.invested = torch.zeros(self.N, dtype=self.dtype)  # dollars spent incl. buy fees

        state = self._get_state()
        if self.save_history:
            self.history = StateHistory()
            self.history.append(state)
        
        return state

    def _trade(self, a: torch.Tensor | None) -> None:
        """
        Rebalance the portfolio toward target value weights.

        Parameters
        ----------
        a : torch.Tensor
            Either:
              - target weights x_star ∈ Δ^{N+1} (cash + N assets), or
              - raw scores/logits which will be softmaxed here (see comment below).
        """
        # 1) current and target weights
        x_curr = self._compute_value_weights()  # [N+1], sums ~ 1
        x_star = torch.softmax(a.to(self.dtype) / self.temp, dim=-1) if a is not None else x_curr

        # 2) turnover for full move x_curr -> x_star
        to_full = 0.5 * (x_star - x_curr).abs().sum() # scalar

        # dead-zone: if turnover is tiny, skip trading altogether
        if to_full.item() < self.transaction_eps:
            # no trades, realized PnL/cost for this step = 0
            self.realized_pnl = torch.zeros(self.N, dtype=self.dtype)
            self.realized_cost = torch.zeros(self.N, dtype=self.dtype)
            return x_star - x_curr

        # 3) turnover cap: at most kappa fraction of portfolio value trades
        kappa = 0.5 # e.g. 10% of portfolio per step
        lam = min(1.0, float(kappa / (to_full.item() + self.eps)))

        x_exec = (1.0 - lam) * x_curr + lam * x_star  # [N+1]

        # 4) translate executed weights into dollar deltas
        V = float(self.V)
        dollar_curr_assets = self.w * self.p  # [N]
        dollar_targ_assets = x_exec[1:] * V
        delta = dollar_targ_assets - dollar_curr_assets  # + buy, - sell

        # reset realized PnL/cost for this step
        self.realized_pnl  = torch.zeros(self.N, dtype=self.dtype)
        self.realized_cost = torch.zeros(self.N, dtype=self.dtype)

        # ===== Sells first (delta < 0) =====
        sell_idx = delta < 0
        if sell_idx.any():
            sell_notional = (-delta[sell_idx]).clamp(min=0.0)  # dollars to free by selling

            # pre-trade snapshots for average-cost basis
            w_before        = self.w.clone()
            invested_before = self.invested.clone()

            p_sel      = self.p[sell_idx]
            w_sel      = w_before[sell_idx]
            inv_sel    = invested_before[sell_idx]

            # units to sell, limited by position size
            units_sold = (sell_notional / p_sel).clamp(max=w_sel)
            notional   = units_sold * p_sel
            fees_s     = self.s_fee[sell_idx] * notional
            proceeds   = notional - fees_s  # net cash inflow

            # average cost per unit BEFORE trade
            avg_cost_sel = torch.zeros_like(inv_sel)
            has_pos_sel = w_sel > self.eps
            avg_cost_sel[has_pos_sel] = inv_sel[has_pos_sel] / w_sel[has_pos_sel]

            base_sel = units_sold * avg_cost_sel  # cost basis of sold units

            # update state: cash, positions, invested
            self.C = self.C + proceeds.sum()

            self.w[sell_idx] = w_sel - units_sold
            self.invested[sell_idx] = (inv_sel - base_sel).clamp(min=0.0)

            # zero out dust positions
            dust_mask = self.w <= self.eps
            if torch.any(dust_mask):
                self.w[dust_mask] = 0.0
                self.invested[dust_mask] = 0.0

            # realized PnL/cost on sold legs
            self.realized_pnl[sell_idx]  = (proceeds - base_sel)
            self.realized_cost[sell_idx] = base_sel + fees_s

        # ===== Then buys (delta > 0) =====
        buy_idx = delta > 0
        if buy_idx.any():
            buy_notional = delta[buy_idx]
            p_buy        = self.p[buy_idx]

            # cap total buy by available cash
            total_desired = buy_notional.sum()
            cash = self.C
            scale = min(1.0, float(cash / (total_desired + 1e-12)))
            buy_notional = buy_notional * scale

            fee_b = self.b_fee[buy_idx] * buy_notional
            effective = buy_notional - fee_b
            units_bought = effective / p_buy.clamp(min=1e-12)

            self.w[buy_idx] += units_bought
            self.C          -= buy_notional.sum()

            # dollars invested include buy fees (you pay them)
            self.invested[buy_idx] += buy_notional

        return x_star - x_curr # for debugging / analysis
        
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

        # update smoothed portfolio value
        alpha_V = 1 - torch.exp(-self.dt / self.tau_p)
        self.V_smooth += alpha_V * (self.V - self.V_smooth)

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
                roi = (self.realized_pnl / (self.realized_cost + self.eps))
                reward = float(roi.sum().detach()) # sum over assets (simple, scale-free)

        elif self.reward_mode == "smooth_roi":
            smooth_roi = (self.V - self.V_smooth) / (self.V_smooth)
            reward = smooth_roi.mean().item() # average over τ-grid
        
        else:
            raise NotImplementedError(f"The reward mode {self.reward_mode} is not implemented.")
        
        return reward

    def step(self, a: torch.Tensor | None, data: dict) -> Tuple[State, float, bool, Dict]:
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
        action = self._trade(a)

        # advance to new state using data
        self._update(data)

        # calculate reward signal
        reward = self._reward()

        # termination & bookkeeping
        done = float(self.V) <= self.bankruptcy_threshold
        if done:
            reward = -100.0
        info = {
            "t": self.t,
            "a": action.tolist(),
            "C": float(self.C),
            "V": float(self.V),
            "w": self.w.tolist(),
            "p": self.p.tolist(),
        }

        # fetch new state and store in hos
        next_state = self._get_state()
        if self.save_history and self.history is not None:
            self.history.append(next_state)

        return next_state, reward, done, info
