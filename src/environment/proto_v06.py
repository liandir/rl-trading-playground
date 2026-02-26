from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch

PI = 3.141592653589793238462


@dataclass
class State:
    time: torch.Tensor            # [6]    (cyclic time features)
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

    def __init__(
        self,
        N: int,
        C0: float,
        tau_p: torch.Tensor,
        *,
        # optional
        temp: float = 1.0,
        bankruptcy_threshold: float = 1.0,
        transaction_eps: float = 1.0,
        use_dollar_volume: bool = True,
        save_history: bool = False,
        sell_fee: float = 1.0,
        buy_fee: float = 1.0,
        tax_rate: float = 0.26,
        min_buy_dollars: float = 10.0,
        dV_coeff: float = 0.02,
        # numerics
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-12,
    ):
        # shape checks
        assert isinstance(N, int) and N > 0
        assert tau_p.ndim == 1 and tau_p.numel() > 0
        assert min_buy_dollars >= buy_fee, "min_buy_dollars must be greater than buy_fee to allow any buys"

        # basic properties
        self.N = N
        self.dtype = dtype
        self.eps = float(eps)

        # controls
        self.temp = float(temp)
        self.bankruptcy_threshold = float(bankruptcy_threshold)
        self.use_dollar_volume = bool(use_dollar_volume)
        self.transaction_eps = float(transaction_eps)
        self.tax_rate = tax_rate
        self.min_buy_dollars = float(min_buy_dollars)

        # reward shaping for realized_roi
        self.dV_coeff = dV_coeff

        # fees (scalar → broadcast)
        self.s_fee = sell_fee
        self.b_fee = buy_fee

        # τ-grid for price/volume features
        self.tau_p = tau_p.to(self.dtype)
        self.M = self.tau_p.numel()

        # history storage
        self.save_history = save_history
        self.history = StateHistory() if save_history else None

        # initial capital (wallets start at zero)
        self.C0 = float(C0)

        # state/action sizes
        self.state_dim = self.M*N + 4*N + 8
        self.action_dim = N + 2
        
    def _compute_time_vector(self) -> torch.Tensor:
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
        # get cash fraction
        x_cash = (self.C / self.V)[None]

        # get fraction of each asset
        x_assets = (self.w * self.p) / self.V
        
        return torch.concat([x_cash, x_assets])
    
    def _compute_cost_weights(self):
        S = self.invested.sum()
        c_rel = (self.invested / (S + self.eps)) if S.item() > 0 else torch.zeros(self.N, dtype=self.dtype)
        rho = S / (S + self.C)
        return c_rel, rho
    
    def _compute_moneyness(self):
        has_pos = self.w > self.eps
        avg_cost = torch.zeros_like(self.invested)
        avg_cost[has_pos] = self.invested[has_pos] / self.w[has_pos]
        m = torch.zeros(self.N, dtype=self.dtype)
        m[has_pos] = (self.p[has_pos] - avg_cost[has_pos]) / (avg_cost[has_pos] + self.eps)
        return m
    
    def _avg_cost_per_unit(self) -> torch.Tensor:
        avg = torch.zeros(self.N, dtype=self.dtype)
        has = self.w > self.eps
        avg[has] = self.invested[has] / self.w[has]
        return avg

    def _unrealized_after_tax_pnl(self) -> torch.Tensor:
        has = self.w > self.eps
        avg_cost = self._avg_cost_per_unit()

        unrl_pre = torch.zeros(self.N, dtype=self.dtype)
        unrl_pre[has] = (self.p[has] - avg_cost[has]) * self.w[has]

        unrl_tax = unrl_pre.clamp(min=0.0) * self.tax_rate
        return unrl_pre - unrl_tax

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
    def V(self) -> torch.Tensor:
        return self.C + torch.sum(self.w * self.p)

    def reset(self, data: dict, C0: Optional[float] = None) -> State:
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
        # self.V_smooth = self.V.repeat(self.M).clone()

        # average-cost-basis bookkeeping for realized PnL
        self.invested = torch.zeros(self.N, dtype=self.dtype) # dollars spent incl. buy fees
        
        # realized PnL bookkeeping (per step)
        self.realized_cost = torch.zeros(self.N, dtype=self.dtype)
        self.realized_pnl  = torch.zeros(self.N, dtype=self.dtype)

        state = self._get_state()
        if self.save_history:
            self.history = StateHistory()
            self.history.append(state)
        
        return state
        
    def _update(self, data: dict) -> None:
        self.V_prev = self.V.detach().clone()

        # ingest time data
        t_new = float(data["time"])
        if t_new < self.t:
            print("WARNING: dt < 0 - setting to zero!")
        self.dt = (t_new - self.t) * float(t_new > self.t)
        self.t = t_new

        # ingest price data
        self.p[:] = data["prices"].to(self.dtype)
        # alpha_p = 1 - torch.exp(-self.dt / self.tau_p)[:, None]
        alpha_p = (self.dt / self.tau_p)[:, None]
        self.p_smooth += alpha_p * (self.p[None, :] - self.p_smooth)
        self.p_rel = (self.p[None, :] - self.p_smooth) / self.p_smooth.clamp(min=self.eps)

        # ingest volume data
        self.v_prev[:] = self.v.clone()
        self.v[:] = data["volume"].to(self.dtype)
        if self.use_dollar_volume:
            self.v[:] *= self.p
        self.v_rel[:] = torch.log((self.v + self.eps) / (self.v_prev + self.eps))

        # update smoothed portfolio value
        # alpha_V = 1 - torch.exp(-self.dt / self.tau_p)
        # alpha_V = self.dt / self.tau_p
        # self.V_smooth += alpha_V * (self.V - self.V_smooth)

    def _trade(self, a: torch.Tensor | None) -> torch.Tensor:
        # reset realized step stats
        self.realized_cost.zero_()
        self.realized_pnl.zero_()

        # compute target weights and translate into dollar deltas
        x_curr = self._compute_value_weights()  # [N+1]
        if a is None:
            x_star = x_curr
        else:
            a = a.to(self.dtype)
            intensity = torch.sigmoid(a[0])
            x_tilted = torch.softmax(a[1:] / self.temp, dim=-1)
            x_star = (1 - intensity) * x_curr + intensity * x_tilted

        # translate executed weights into dollar deltas
        dollar_curr_assets = self.w * self.p            # [N]
        dollar_targ_assets = x_star[1:] * float(self.V) # [N]
        delta = dollar_targ_assets - dollar_curr_assets
        
        # after computing delta
        buy_delta  = torch.clamp(delta, min=0.0)
        sell_delta = torch.clamp(delta, max=0.0)

        # zero small transactions to avoid noise and fee traps
        buy_delta = torch.where(buy_delta < self.min_buy_dollars, torch.zeros_like(buy_delta), buy_delta)
        delta = sell_delta + buy_delta

        # 1) execute sells first to free up cash
        sells = delta < -self.transaction_eps
        if sells.any():
            w_before = self.w[sells].clone()

            # dollars we *want* to reduce, capped by current position value
            desired_sell_dollars = (-delta[sells]).clamp(min=0.0)
            max_sell_dollars = (w_before * self.p[sells]).clamp(min=0.0)
            sell_dollars = torch.minimum(desired_sell_dollars, max_sell_dollars)

            # drop sells that don't cover fixed fee (otherwise selling reduces cash)
            sell_dollars = torch.where(sell_dollars > self.s_fee, sell_dollars, torch.zeros_like(sell_dollars))
            sell_units = sell_dollars / self.p[sells]
            sell_units = torch.minimum(sell_units, w_before)

            # fraction of position sold
            f = (sell_units / w_before).clamp(0.0, 1.0)

            # transaction fees
            fee_dollars = torch.where(sell_dollars > 0.0, self.s_fee, torch.zeros_like(sell_dollars))

            # realized cost basis removed (average cost)
            realized_cost = f * self.invested[sells]

            # proceeds before tax (fees reduce proceeds)
            proceeds = (sell_dollars - fee_dollars).clamp(min=0.0)

            # pre-tax pnl and taxes only on gains
            pre_tax_pnl = proceeds - realized_cost
            tax_dollars = pre_tax_pnl.clamp(min=0.0) * self.tax_rate

            net_proceeds = proceeds - tax_dollars  # cash you receive
            realized_pnl = net_proceeds - realized_cost

            self.C += net_proceeds.sum()
            self.w[sells] -= sell_units
            self.invested[sells] -= realized_cost

            self.realized_cost[sells] = realized_cost
            self.realized_pnl[sells]  = realized_pnl

        # 2) then execute buys with remaining cash (GUARANTEED cash-safe)
        buys = delta > self.transaction_eps
        if buys.any():
            buy_dollars = delta[buys].clone()          # intended pre-fee dollars into assets

            # fixed fee only charged when we actually buy
            fee_dollars = torch.where(buy_dollars > self.eps, self.b_fee, torch.zeros_like(buy_dollars))
            total_cost = buy_dollars + fee_dollars

            # --- GLOBAL CASH GUARANTEE with FIXED per-asset fees ---
            buy_sum = buy_dollars.sum()
            fee_sum = fee_dollars.sum()

            if self.C < fee_sum or buy_sum < 0.0:
                buy_dollars.zero_()
                fee_dollars.zero_()
                total_cost = buy_dollars + fee_dollars
            else:
                # Scale ONLY notionals to fit remaining cash after fixed fees
                scale = torch.minimum(
                    torch.tensor(1.0, dtype=self.dtype),
                    (self.C - fee_sum) / buy_sum
                )
                buy_dollars = buy_dollars * scale

                # If scaling makes some buys too small, drop them and don't charge their fee
                keep = buy_dollars >= self.min_buy_dollars
                buy_dollars = torch.where(keep, buy_dollars, torch.zeros_like(buy_dollars))
                fee_dollars = torch.where(keep, fee_dollars, torch.zeros_like(fee_dollars))

                total_cost = buy_dollars + fee_dollars

            self.C -= total_cost.sum()
            self.w[buys] += buy_dollars / self.p[buys]

            self.invested[buys] += total_cost

        # 4) zero out tiny positions to avoid noise buildup
        dust = (self.w * self.p) < self.transaction_eps
        self.w[dust] = 0.0
        self.invested[dust] = 0.0

        # hard numerical guard
        self.C = torch.clamp(self.C, min=0.0)

        return delta # for debugging / analysis

    def _reward(self):
        # --- realized component (ROI on what was closed this step), weighted by size ---
        if self.realized_cost.sum() > 0.0:
            roi = self.realized_pnl.sum() / self.realized_cost.sum()
            realized_term = roi.item()
        else:
            realized_term = 0.0

        # --- unrealized carry (dense signal), tax-aware ---
        # unrl_after = self._unrealized_after_tax_pnl().sum()
        # carry_term = (unrl_after / self.V).detach().item()

        dV_term = ((self.V - self.V_prev) / self.V_prev).detach().item()
        # dV_term = torch.log(self.V / self.V_prev).detach().item()

        return float(realized_term + self.dV_coeff * dV_term)

    def step(self, a: torch.Tensor | None, data: dict) -> Tuple[State, float, bool, Dict]:
        # trade using action a
        action = self._trade(a)

        # advance to new state using data
        self._update(data)

        # calculate reward signal
        reward = self._reward()

        # termination & bookkeeping
        done = float(self.V) <= self.bankruptcy_threshold
        if done:
            reward -= 10.0
        
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
