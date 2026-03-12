from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch

PI = 3.141592653589793238462


@dataclass
class State:
    time: torch.Tensor            # [6]    cyclic time features
    p_rel: torch.Tensor           # [M, N]
    x_rel: torch.Tensor           # [N+1]  exposure: cash + value weights
    c_rel: torch.Tensor           # [N]    invested weights, stable
    rho: torch.Tensor             # []     overall commitment
    mny: torch.Tensor             # [N]    moneyness vs entry
    v_rel: torch.Tensor           # [N] or [M_u, N] depending on design

    def to_tensor(self):
        return torch.cat([
            self.time.flatten(),
            self.p_rel.flatten(),
            self.x_rel.flatten(),
            self.c_rel.flatten(),
            self.rho.view(1),
            self.mny.flatten(),
            self.v_rel.flatten(),
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
    """
    Discrete-action multi-asset trading environment.

    Action space:
        0                                   -> hold
        1 .. N*K                            -> buy  asset i with bucket k
        N*K+1 .. 2*N*K                      -> sell asset i with bucket k

    where:
        K = len(size_buckets)
        size_buckets default to (0.10, 0.25, 0.50, 1.00)

    Buy semantics:
        spend bucket_fraction * current cash on selected asset,
        subject to fixed buy fee and minimum buy dollars.

    Sell semantics:
        sell bucket_fraction * current units of selected asset,
        subject to fixed sell fee and tax on realized gains.

    Invalid trades:
        If an action cannot be executed meaningfully (too little cash, no position,
        proceeds <= fee, below minimum buy size, etc.), a penalty is applied.

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
        tau_p_live: Optional[float] = None,
        val_coeff: float = 0.1,
        roi_coeff: float = 1.0,
        reward_mode: str = "log",
        size_buckets: Tuple[float, ...] = (0.50, 1.00),
        invalid_trade_penalty: float = 0.0,
        done_reward_penalty: float = 10.0,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ):
        assert isinstance(N, int) and N > 0
        assert tau_p.ndim == 1 and tau_p.numel() > 0
        assert min_buy_dollars >= buy_fee, "min_buy_dollars must be >= buy_fee"
        assert len(size_buckets) > 0, "size_buckets must not be empty"
        assert all(0.0 < x <= 1.0 for x in size_buckets), "all size buckets must be in (0, 1]"
        assert reward_mode in ("return", "log"), "reward_mode must be one of ('return', 'log')"
        if tau_p_live is not None:
            assert tau_p_live > 0.0, "tau_p_live must be > 0"
            assert bool(torch.all(tau_p_live < tau_p).item()), "tau_p_live must be smaller than all tau_p"

        self.N = N
        self.dtype = dtype
        self.eps = float(eps)

        self.bankruptcy_threshold = float(bankruptcy_threshold)
        self.transaction_eps = float(transaction_eps)
        self.use_dollar_volume = bool(use_dollar_volume)
        self.tax_rate = float(tax_rate)
        self.min_buy_dollars = float(min_buy_dollars)
        self.tau_p_live = float(tau_p_live) if tau_p_live is not None else None
        self.val_coeff = float(val_coeff)
        self.roi_coeff = float(roi_coeff)
        self.reward_mode = reward_mode
        self.invalid_trade_penalty = float(invalid_trade_penalty)
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

        # state = time(6) + p_rel(M*N) + x_rel(N+1) + c_rel(N) + rho(1) + mny(N) + v_rel(N)
        self.state_dim = self.M * N + 4 * N + 8

        # discrete actions: hold + buys + sells
        self.action_dim = 1 + 2 * self.N * self.K

        # runtime attributes initialized in reset()
        self.C = None
        self.w = None
        self.t = None
        self.dt = None
        self.p = None
        self.v = None
        self.v_prev = None
        self.p_smooth = None
        self.p_rel = None
        self.v_rel = None
        self.V_prev = None
        self.invested = None
        self.realized_cost = None
        self.realized_pnl = None

    # -------------------------------------------------------------------------
    # action helpers
    # -------------------------------------------------------------------------

    def encode_hold(self) -> int:
        return 0

    def encode_buy(self, asset_idx: int, bucket_idx: int) -> int:
        if not (0 <= asset_idx < self.N):
            raise ValueError("asset_idx out of range")
        if not (0 <= bucket_idx < self.K):
            raise ValueError("bucket_idx out of range")
        return 1 + asset_idx * self.K + bucket_idx

    def encode_sell(self, asset_idx: int, bucket_idx: int) -> int:
        if not (0 <= asset_idx < self.N):
            raise ValueError("asset_idx out of range")
        if not (0 <= bucket_idx < self.K):
            raise ValueError("bucket_idx out of range")
        return 1 + self.N * self.K + asset_idx * self.K + bucket_idx

    def decode_action(self, action_idx: int) -> tuple[str, int | None, float | None]:
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

        angles = torch.tensor([period_day, period_week, period_year], dtype=self.dtype)
        return torch.cat([torch.sin(2 * PI * angles), torch.cos(2 * PI * angles)])

    def _compute_value_weights(self) -> torch.Tensor:
        x_cash = (self.C / self.V)[None]
        x_assets = (self.w * self.p) / self.V
        return torch.cat([x_cash, x_assets])

    def _compute_cost_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        S = self.invested.sum()
        c_rel = (self.invested / (S + self.eps)) if S.item() > 0 else torch.zeros(self.N, dtype=self.dtype)
        rho = S / (S + self.C + self.eps)
        return c_rel, rho

    def _compute_moneyness(self) -> torch.Tensor:
        has_pos = self.w > self.eps
        avg_cost = torch.zeros_like(self.invested)
        avg_cost[has_pos] = self.invested[has_pos] / self.w[has_pos].clamp(min=self.eps)

        m = torch.zeros(self.N, dtype=self.dtype)
        m[has_pos] = (self.p[has_pos] - avg_cost[has_pos]) / (avg_cost[has_pos] + self.eps)
        return m

    def _avg_cost_per_unit(self) -> torch.Tensor:
        avg = torch.zeros(self.N, dtype=self.dtype)
        has = self.w > self.eps
        avg[has] = self.invested[has] / self.w[has].clamp(min=self.eps)
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
        x_rel = self._compute_value_weights().to(self.dtype)
        c_rel, rho = self._compute_cost_weights()
        mny = self._compute_moneyness()

        return State(
            time=t_vec,
            p_rel=self.p_rel.clone(),
            v_rel=self.v_rel.clone(),
            x_rel=x_rel,
            c_rel=c_rel,
            rho=rho,
            mny=mny,
        )

    # -------------------------------------------------------------------------
    # portfolio properties
    # -------------------------------------------------------------------------

    @property
    def V(self) -> torch.Tensor:
        return self.C + torch.sum(self.w * self.p)

    # -------------------------------------------------------------------------
    # reset / update
    # -------------------------------------------------------------------------

    def reset(self, data: dict, C0: Optional[float] = None) -> State:
        if C0 is not None:
            self.C0 = float(C0)

        self.C = torch.tensor(self.C0, dtype=self.dtype)
        self.w = torch.zeros(self.N, dtype=self.dtype)

        self.t = float(data["time"])
        self.dt = 0.0
        self.p = data["prices"].to(self.dtype).clone()
        self.v = data["volume"].to(self.dtype).clone()
        if self.use_dollar_volume:
            self.v *= self.p

        self.v_prev = self.v.clone()

        self.p_smooth = self.p[None, :].repeat(self.M, 1).clone()
        self.p_rel = torch.zeros((self.M, self.N), dtype=self.dtype)
        self.v_rel = torch.zeros((self.N,), dtype=self.dtype)

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
        self.V_prev = self.V.detach().clone()

        t_new = float(data["time"])
        if t_new < self.t:
            print("WARNING: dt < 0 - setting to zero!")

        self.dt = (t_new - self.t) * float(t_new > self.t)
        self.t = t_new

        p_new = data["prices"].to(self.dtype)
        if self.tau_p_live is None:
            self.p[:] = p_new
        else:
            alpha_live = 1 - torch.exp(torch.tensor(-self.dt / self.tau_p_live, dtype=self.dtype))
            self.p += alpha_live * (p_new - self.p)

        alpha_p = 1 - torch.exp(-self.dt / self.tau_p.clamp(min=self.eps))[:, None]
        self.p_smooth += alpha_p * (self.p[None, :] - self.p_smooth)
        self.p_rel = (self.p[None, :] - self.p_smooth) / self.p_smooth.clamp(min=self.eps)

        self.v_prev[:] = self.v.clone()
        self.v[:] = data["volume"].to(self.dtype)
        if self.use_dollar_volume:
            self.v[:] *= self.p

        self.v_rel[:] = torch.log((self.v + self.eps) / (self.v_prev + self.eps))

    # -------------------------------------------------------------------------
    # trading
    # -------------------------------------------------------------------------

    def _trade(self, a: int | torch.Tensor | None) -> tuple[torch.Tensor, bool]:
        """
        Returns:
            action_info: tensor [action_type, asset_idx, fraction]
                action_type: 0=hold, 1=buy, 2=sell
                asset_idx:  -1 for hold
                fraction:   0 for hold, else chosen size bucket
            valid_trade: bool
                True if the action is hold or a trade was actually executed.

        Notes:
            - self.eps is used only for numerical safety (division / clamping).
            - economic validity is determined by explicit thresholds:
                * buy fee
                * sell fee
                * min_buy_dollars
                * available cash / held units
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

    def _reward(self, valid_trade: bool) -> float:
        if self.realized_cost.sum() > 0.0:
            reward = self.roi_coeff * (self.realized_pnl.sum() / self.realized_cost.sum()).item()
        else:
            if self.reward_mode == "return":
                reward = self.val_coeff * ((self.V - self.V_prev) / self.V_prev).detach().item()
            else:
                reward = self.val_coeff * torch.log(self.V / self.V_prev).detach().item()

        if not valid_trade:
            reward -= self.invalid_trade_penalty

        return float(reward)

    # -------------------------------------------------------------------------
    # main step
    # -------------------------------------------------------------------------

    def step(self, a: int | torch.Tensor | None, data: dict) -> Tuple[State, float, bool, Dict]:
        action_info, valid_trade = self._trade(a)
        self._update(data)
        reward = self._reward(valid_trade)

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
