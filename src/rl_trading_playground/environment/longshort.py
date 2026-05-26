r"""Pure-discrete long/short multi-currency trading environment (no leverage).

Discrete counterpart to
[rl_trading_playground.environment.generic.longshort](../generic/longshort.html): the
hybrid action's continuous fraction is removed, and every open commits
*all* available cash as collateral. Positions are signed
($u_k > 0$ long, $u_k < 0$ short), shorts collateralised 1:1.

State space
===========

The flat state vector has dimension

$$
\dim(\mathcal{S}) = 3 M N + 5 N + 8,
$$

$$
s_t = \big[\, t_{\text{vec}},\; p_{\text{rel}},\; \eta,\;
x_{\text{rel}},\; c_{\text{rel}},\; \rho_t,\; v_{\text{rel}},\;
m,\; \sigma,\; \text{vol}_{\text{rel}} \,\big].
$$

Shapes:

$$
t_{\text{vec}} \in \mathbb{R}^{6},\;
p_{\text{rel}}, v_{\text{rel}}, \text{vol}_{\text{rel}} \in \mathbb{R}^{M\times N},\;
\eta, c_{\text{rel}}, m, \sigma \in \mathbb{R}^{N},\;
x_{\text{rel}} \in \mathbb{R}^{N+1},\;
\rho_t \in \mathbb{R}.
$$

Per-asset value weight is *signed*:

$$
x_{\text{rel}, k+1} = \frac{u_k\, p_k}{V_t},\qquad
V_t = C_t + \sum_k u_k p_k,
$$

with $x_{\text{rel}, 0} = C_t / V_t$. The committed distribution
$c_{\text{rel}, k} = I_k / (S_t + \varepsilon) \ge 0$ and overall
commitment $\rho_t = S_t / (S_t + C_t)$ use posted collateral
$I_k = $ cash committed at open time. Position side is

$$
\sigma_k =
\begin{cases}
+1 & u_k > 0,\\
0 & u_k = 0,\\
-1 & u_k < 0,
\end{cases}
$$

and signed unrealised after-tax PnL relative to committed cash is

$$
m_k =
\begin{cases}
\dfrac{u_k(p_k - \bar p^{\text{ent}}_k) - \tau\,[u_k(p_k - \bar p^{\text{ent}}_k)]_+ - \phi_{\text{c}}}{I_k}
& \text{open},\\[1.2ex]
0 & \text{flat}.
\end{cases}
$$

See [discrete.py](discrete.html) for the multi-scale price, volume and
volatility definitions.

Action space
============

Discrete head only:

$$
a \in \{0, 1, \dots, 3N\},\qquad |\mathcal{A}| = 1 + 3N,
$$

$$
a =
\begin{cases}
0 & \text{hold},\\
1 \dots N & \text{open LONG asset } k = a - 1,\\
N+1 \dots 2N & \text{open SHORT asset } k = a - 1 - N,\\
2N+1 \dots 3N & \text{CLOSE position } k = a - 1 - 2N.
\end{cases}
$$

Every open commits the full available cash $C_t$ as collateral. Flips
auto-close the existing opposite position first. Close is reduce-only.

Reward
======

Identical to the hybrid env:

$$
r_t =
\begin{cases}
\lambda_{\text{roi}} \dfrac{\sum_k \text{PnL}^{\text{real}}_k}
                           {\sum_k \text{Cost}^{\text{real}}_k}
& \sum_k \text{Cost}^{\text{real}}_k > \varepsilon,\\[1.4ex]
\lambda_{\text{val}}\,\log(V_t / V_{t-1}) & \text{reward\_mode = \texttt{log}},\\[0.3ex]
\lambda_{\text{val}}\,(V_t - V_{t-1})/V_{t-1} & \text{reward\_mode = \texttt{return}}.
\end{cases}
$$

Bankruptcy at $V_t \le V_{\text{bk}}$ ends the episode and adds the
penalty $-\lambda_{\text{done}}$.
"""
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
    hl_rel: torch.Tensor          # [N]
    x_rel: torch.Tensor           # [N+1]  signed value weights: cash + pos_units*p / V
    c_rel: torch.Tensor           # [N]    committed distribution over assets (>=0)
    rho: torch.Tensor             # []     overall commitment fraction
    v_rel: torch.Tensor           # [M, N]
    unrl_rel: torch.Tensor        # [N]    signed after-tax PnL / committed (0 when flat)
    side_rel: torch.Tensor        # [N]    -1 short, 0 flat, +1 long
    vol_rel: torch.Tensor         # [M, N]

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
            self.side_rel.flatten(),
            self.vol_rel.flatten(),
        ])


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

    def get_side_rel(self) -> torch.Tensor:
        """Return the side rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return torch.stack([s.side_rel for s in self.states])

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


class LongShortEnv:
    """
    Discrete-action multi-asset long/short trading environment.

    Positions are signed: long > 0, short < 0. Shorts commit cash as collateral
    at 1:1 (no leverage) — equivalent to a linear perpetual with leverage=1.

    Action space (action_dim = 1 + 3*N):
        0              -> hold
        1 .. N         -> open LONG  asset k (auto-closes existing short on k)
        N+1 .. 2N      -> open SHORT asset k (auto-closes existing long on k)
        2N+1 .. 3N     -> CLOSE position on asset k (reduce-only)

    Each open commits all available cash as collateral on the target asset.

    State features:
        time     [6]
        p_rel    [M, N]
        hl_rel   [N]
        x_rel    [N+1]   signed value weights (cash + pos_units*p / V)
        c_rel    [N]     committed distribution (>=0)
        rho      [1]     total committed / (total committed + cash)
        v_rel    [M, N]
        unrl_rel [N]     signed after-tax PnL / committed
        side_rel [N]     -1 short / 0 flat / +1 long
        vol_rel  [M, N]

    state_dim = 3*M*N + 5*N + 8
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
            dtype (torch.dtype): The dtype value. Defaults to ``torch.float32``.
            eps (float): The eps value. Defaults to ``1e-08``.

        Returns:
            None: This function does not return a value.
        """
        assert isinstance(N, int) and N > 0
        assert tau_p.ndim == 1 and tau_p.numel() > 0
        assert min_open_dollars >= open_fee, "min_open_dollars must be >= open_fee"
        assert reward_mode in ("return", "log"), "reward_mode must be one of ('return', 'log')"

        self.N = N
        self.dtype = dtype
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

        self.tau_p = tau_p.to(self.dtype)
        self.M = self.tau_p.numel()

        self.save_history = bool(save_history)
        self.history = StateHistory() if self.save_history else None

        self.C0 = float(C0)

        self.state_dim = 3 * self.M * N + 5 * N + 8
        self.action_dim = 1 + 3 * self.N

        # runtime attributes
        self.C = None
        self.pos_units = None        # signed
        self.entry_price = None
        self.committed = None
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
        self.realized_cost = None
        self.realized_pnl = None

    # -------------------------------------------------------------------------
    # action helpers
    # -------------------------------------------------------------------------

    def encode_hold(self) -> int:
        """Encode hold for LongShortEnv.

        Returns:
            int: The computed or requested result.
        """
        return 0

    def encode_long(self, asset_idx: int) -> int:
        """Encode long for LongShortEnv.

        Args:
            asset_idx (int): The asset idx value.

        Returns:
            int: The computed or requested result.
        """
        if not (0 <= asset_idx < self.N):
            raise ValueError("asset_idx out of range")
        return 1 + asset_idx

    def encode_short(self, asset_idx: int) -> int:
        """Encode short for LongShortEnv.

        Args:
            asset_idx (int): The asset idx value.

        Returns:
            int: The computed or requested result.
        """
        if not (0 <= asset_idx < self.N):
            raise ValueError("asset_idx out of range")
        return 1 + self.N + asset_idx

    def encode_close(self, asset_idx: int) -> int:
        """Encode close for LongShortEnv.

        Args:
            asset_idx (int): The asset idx value.

        Returns:
            int: The computed or requested result.
        """
        if not (0 <= asset_idx < self.N):
            raise ValueError("asset_idx out of range")
        return 1 + 2 * self.N + asset_idx

    def decode_action(self, action_idx: int) -> tuple[str, int | None]:
        """Decode action for LongShortEnv.

        Args:
            action_idx (int): The action idx value.

        Returns:
            tuple[str, int | None]: The computed or requested result.
        """
        if action_idx == 0:
            return ("hold", None)
        if 1 <= action_idx <= self.N:
            return ("long", action_idx - 1)
        if self.N + 1 <= action_idx <= 2 * self.N:
            return ("short", action_idx - 1 - self.N)
        if 2 * self.N + 1 <= action_idx < self.action_dim:
            return ("close", action_idx - 1 - 2 * self.N)
        raise ValueError(f"invalid action index {action_idx}")

    def valid_action_mask(self) -> torch.Tensor:
        """Boolean mask of shape [action_dim]. An open is valid if the agent
        is not already positioned the same way on the target asset, AND
        (after any implicit close of the opposite position) enough cash
        remains to satisfy the fixed open fee + minimum open dollars.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[0] = True  # hold

        cash = float(self.C.item())

        for k in range(self.N):
            units = float(self.pos_units[k].item())
            is_long = units > self.eps
            is_short = units < -self.eps
            is_flat = not (is_long or is_short)

            # close_k ok if notional > close_fee
            notional = abs(units) * float(self.p[k].item())
            close_ok = (not is_flat) and notional > self.c_fee

            # cash available after closing (if any position) — conservative
            if is_flat:
                cash_avail = cash
            elif close_ok:
                committed = float(self.committed[k].item())
                entry = float(self.entry_price[k].item())
                price = float(self.p[k].item())
                pnl = units * (price - entry)
                gross = committed + pnl
                proceeds = max(gross - self.c_fee, 0.0)
                pre_tax = proceeds - committed
                tax = max(pre_tax, 0.0) * self.tax_rate
                net_proceeds = proceeds - tax
                cash_avail = cash + net_proceeds
            else:
                cash_avail = cash  # can't close → can't flip

            can_open = cash_avail > self.o_fee and (cash_avail - self.o_fee) >= self.min_open_dollars

            # long k
            if is_long:
                mask[self.encode_long(k)] = False
            elif is_short:
                mask[self.encode_long(k)] = close_ok and can_open
            else:
                mask[self.encode_long(k)] = can_open

            # short k
            if is_short:
                mask[self.encode_short(k)] = False
            elif is_long:
                mask[self.encode_short(k)] = close_ok and can_open
            else:
                mask[self.encode_short(k)] = can_open

            # close k
            mask[self.encode_close(k)] = close_ok

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
        """Signed value weights: cash / V and pos_units * p / V.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        V = self.V.clamp(min=self.eps)
        x_cash = (self.C / V)[None]
        x_assets = (self.pos_units * self.p) / V
        return torch.cat([x_cash, x_assets])

    def _compute_cost_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the cost weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The computed or requested result.
        """
        S = self.committed.sum()
        c_rel = (self.committed / (S + self.eps)) if S.item() > 0 else torch.zeros(self.N, dtype=self.dtype)
        rho = S / (S + self.C + self.eps)
        return c_rel, rho

    def _compute_unrl_rel(self) -> torch.Tensor:
        """Signed unrealized after-tax PnL divided by committed collateral.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        has = self.committed > self.eps
        unrl_rel = torch.zeros(self.N, dtype=self.dtype)
        if has.any():
            pnl = self.pos_units * (self.p - self.entry_price)           # signed
            tax = pnl.clamp(min=0.0) * self.tax_rate
            net = pnl - tax
            unrl_rel[has] = net[has] / self.committed[has]
        return unrl_rel

    def _compute_side_rel(self) -> torch.Tensor:
        """Compute the side rel.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        side = torch.zeros(self.N, dtype=self.dtype)
        side[self.pos_units > self.eps] = 1.0
        side[self.pos_units < -self.eps] = -1.0
        return side

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
        for key in ("close", "high", "low", "volume", "time"):
            if key not in data:
                raise KeyError(f"data must contain '{key}'")

        close  = data["close"].to(self.dtype).reshape(-1)
        high   = data["high"].to(self.dtype).reshape(-1)
        low    = data["low"].to(self.dtype).reshape(-1)
        volume = data["volume"].to(self.dtype).reshape(-1)

        for name, value in (("close", close), ("volume", volume), ("high", high), ("low", low)):
            if value.numel() != self.N:
                raise ValueError(f"data['{name}'] must have {self.N} elements, got {value.numel()}")

        return float(data["time"]), close, high, low, volume

    def _get_state(self) -> State:
        """Return the current state snapshot.

        Returns:
            State: The computed or requested result.
        """
        t_vec = self._compute_time_vector().to(self.dtype)
        x_rel = self._compute_value_weights().to(self.dtype)
        c_rel, rho = self._compute_cost_weights()
        unrl_rel = self._compute_unrl_rel()
        side_rel = self._compute_side_rel()

        return State(
            time=t_vec,
            p_rel=self.p_rel.clone(),
            hl_rel=self.hl_rel.clone(),
            v_rel=self.v_rel.clone(),
            x_rel=x_rel,
            c_rel=c_rel,
            rho=rho,
            unrl_rel=unrl_rel,
            side_rel=side_rel,
            vol_rel=self.vol_rel.clone(),
        )

    # -------------------------------------------------------------------------
    # portfolio property
    # -------------------------------------------------------------------------

    @property
    def V(self) -> torch.Tensor:
        """Equity = cash + sum(committed + signed PnL).

        Returns:
            torch.Tensor: The computed or requested result.
        """
        pnl = self.pos_units * (self.p - self.entry_price)
        return self.C + self.committed.sum() + pnl.sum()

    # -------------------------------------------------------------------------
    # reset / update
    # -------------------------------------------------------------------------

    def reset(self, data: dict, C0: Optional[float] | None = None) -> State:
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
        self.pos_units   = torch.zeros(self.N, dtype=self.dtype)
        self.entry_price = torch.zeros(self.N, dtype=self.dtype)
        self.committed   = torch.zeros(self.N, dtype=self.dtype)

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

        self.realized_cost = torch.zeros(self.N, dtype=self.dtype)
        self.realized_pnl  = torch.zeros(self.N, dtype=self.dtype)

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
    # trading primitives
    # -------------------------------------------------------------------------

    def _close_position(self, k: int) -> bool:
        """Close position on asset k. Returns True if a close was executed.

        Args:
            k (int): The k value.

        Returns:
            bool: The computed or requested result.
        """
        units = self.pos_units[k]
        if abs(float(units.item())) <= self.eps:
            return False

        price = self.p[k]
        notional = torch.abs(units) * price
        if float(notional.item()) <= self.c_fee:
            return False

        committed_k = self.committed[k].clone()
        entry_k = self.entry_price[k].clone()
        pnl = units * (price - entry_k)
        gross = committed_k + pnl

        proceeds = (gross - self.c_fee).clamp(min=0.0)
        pre_tax_pnl = proceeds - committed_k
        tax = pre_tax_pnl.clamp(min=0.0) * self.tax_rate
        net_proceeds = proceeds - tax
        realized_pnl = net_proceeds - committed_k

        self.C += net_proceeds
        self.pos_units[k]   = torch.tensor(0.0, dtype=self.dtype)
        self.committed[k]   = torch.tensor(0.0, dtype=self.dtype)
        self.entry_price[k] = torch.tensor(0.0, dtype=self.dtype)

        self.realized_cost[k] += committed_k
        self.realized_pnl[k]  += realized_pnl
        return True

    def _open_position(self, k: int, side: int) -> bool:
        """Open a side=+1 long or side=-1 short on asset k using all available cash.

        Args:
            k (int): The k value.
            side (int): The side value.

        Returns:
            bool: The computed or requested result.
        """
        cash = float(self.C.item())
        buy_dollars = cash - self.o_fee
        if not (cash > self.o_fee and buy_dollars >= self.min_open_dollars):
            return False

        price_k = self.p[k].clamp(min=self.eps)
        buy_dollars_t = torch.tensor(buy_dollars, dtype=self.dtype)
        open_fee_t = torch.tensor(self.o_fee, dtype=self.dtype)

        units_mag = buy_dollars_t / price_k
        total_cost = torch.minimum(buy_dollars_t + open_fee_t, self.C)

        self.C -= total_cost
        self.pos_units[k]   = float(side) * units_mag
        self.committed[k]   = buy_dollars_t
        self.entry_price[k] = self.p[k].clone()
        return True

    def _trade(self, a: int | torch.Tensor | None) -> tuple[torch.Tensor, bool]:
        """Trade for LongShortEnv.

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

        # action_info = [action_type, asset_idx, side]  (side: -1 short, 0 hold/close, +1 long)
        action_info = torch.tensor([0.0, -1.0, 0.0], dtype=self.dtype)

        if action_idx == 0:
            return action_info, True

        kind, k = self.decode_action(action_idx)
        action_info[1] = float(k)

        if kind == "close":
            action_info[0] = 3.0
            did = self._close_position(k)
            return action_info, did

        # long / short open
        side = 1 if kind == "long" else -1
        action_info[0] = 1.0 if side == 1 else 2.0
        action_info[2] = float(side)

        # flip: close existing opposite position first
        units = float(self.pos_units[k].item())
        if (side == 1 and units < -self.eps) or (side == -1 and units > self.eps):
            self._close_position(k)

        # if already same-side, reject the open (already positioned)
        units_after = float(self.pos_units[k].item())
        if (side == 1 and units_after > self.eps) or (side == -1 and units_after < -self.eps):
            return action_info, False

        did = self._open_position(k, side)

        # dust cleanup (safety net after float ops)
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
        """Reward for LongShortEnv.

        Returns:
            float: The computed or requested result.
        """
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
            "action_type": int(action_info[0].item()),   # 0=hold, 1=long, 2=short, 3=close
            "action_asset": int(action_info[1].item()),  # -1 for hold
            "action_side": int(action_info[2].item()),   # -1 short, 0 hold/close, +1 long
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


class BatchedLongShortEnv:
    """
    Fully-batched B-environment version of LongShortEnv.

    Action space and semantics are identical to LongShortEnv.
    Market data is passed as (B, N) tensors per call.

    Observation is a (B, state_dim) tensor with state_dim = 3*M*N + 5*N + 8.
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
            dtype (torch.dtype): The dtype value. Defaults to ``torch.float32``.
            eps (float): The eps value. Defaults to ``1e-08``.

        Returns:
            None: This function does not return a value.
        """
        assert isinstance(B, int) and B > 0
        assert isinstance(N, int) and N > 0
        assert tau_p.ndim == 1 and tau_p.numel() > 0
        assert min_open_dollars >= open_fee
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
        self.min_open_dollars = float(min_open_dollars)
        self.val_coeff = float(val_coeff)
        self.roi_coeff = float(roi_coeff)
        self.reward_mode = reward_mode
        self.done_reward_penalty = float(done_reward_penalty)
        self.o_fee = float(open_fee)
        self.c_fee = float(close_fee)

        self.tau_p = tau_p.to(dtype)
        self.M = self.tau_p.numel()

        self.state_dim = 3 * self.M * N + 5 * N + 8
        self.action_dim = 1 + 3 * N

        self._b_idx = torch.arange(B)

        # runtime state
        self.C = None             # (B,)
        self.pos_units = None     # (B, N) signed
        self.entry_price = None   # (B, N)
        self.committed = None     # (B, N)  >=0
        self.p = None             # (B, N)
        self.v = None             # (B, N)
        self.p_smooth = None      # (B, M, N)
        self.v_smooth = None      # (B, M, N)
        self.hl_rel = None        # (B, N)
        self.hl_smooth = None     # (B, M, N)
        self.p_rel = None         # (B, M, N)
        self.v_rel = None         # (B, M, N)
        self.vol_rel = None       # (B, M, N)
        self.V_prev = None        # (B,)
        self.realized_cost = None # (B, N)
        self.realized_pnl = None  # (B, N)
        self.t = None             # (B,)

    # -------------------------------------------------------------------------
    # action helpers
    # -------------------------------------------------------------------------

    def encode_hold(self) -> int:
        """Encode hold for BatchedLongShortEnv.

        Returns:
            int: The computed or requested result.
        """
        return 0

    def encode_long(self, asset_idx: int) -> int:
        """Encode long for BatchedLongShortEnv.

        Args:
            asset_idx (int): The asset idx value.

        Returns:
            int: The computed or requested result.
        """
        return 1 + asset_idx

    def encode_short(self, asset_idx: int) -> int:
        """Encode short for BatchedLongShortEnv.

        Args:
            asset_idx (int): The asset idx value.

        Returns:
            int: The computed or requested result.
        """
        return 1 + self.N + asset_idx

    def encode_close(self, asset_idx: int) -> int:
        """Encode close for BatchedLongShortEnv.

        Args:
            asset_idx (int): The asset idx value.

        Returns:
            int: The computed or requested result.
        """
        return 1 + 2 * self.N + asset_idx

    # -------------------------------------------------------------------------
    # portfolio property
    # -------------------------------------------------------------------------

    @property
    def V(self) -> torch.Tensor:   # (B,)
        """V for BatchedLongShortEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        pnl = self.pos_units * (self.p - self.entry_price)   # (B, N)
        return self.C + self.committed.sum(dim=-1) + pnl.sum(dim=-1)

    # -------------------------------------------------------------------------
    # reset
    # -------------------------------------------------------------------------

    def reset(
        self,
        close: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        volume: torch.Tensor,
        time: torch.Tensor,
        C0: Optional[float] | None = None,
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

        self.C           = torch.full((B,), self.C0, dtype=self.dtype)
        self.pos_units   = torch.zeros(B, N, dtype=self.dtype)
        self.entry_price = torch.zeros(B, N, dtype=self.dtype)
        self.committed   = torch.zeros(B, N, dtype=self.dtype)
        self.p           = close.clone()
        self.v           = volume.clone()
        if self.use_dollar_volume:
            self.v = self.v * self.p

        self.p_smooth  = self.p[:, None, :].expand(B, M, N).clone()
        self.v_smooth  = self.v[:, None, :].expand(B, M, N).clone()
        self.hl_rel    = self._hl_rel(high, low)
        self.hl_smooth = self.hl_rel[:, None, :].expand(B, M, N).clone()

        self.p_rel   = torch.zeros(B, M, N, dtype=self.dtype)
        self.v_rel   = torch.zeros(B, M, N, dtype=self.dtype)
        self.vol_rel = torch.zeros(B, M, N, dtype=self.dtype)

        self.V_prev        = self.C.clone()
        self.realized_cost = torch.zeros(B, N, dtype=self.dtype)
        self.realized_pnl  = torch.zeros(B, N, dtype=self.dtype)

        return self._obs()

    # -------------------------------------------------------------------------
    # internal helpers
    # -------------------------------------------------------------------------

    def _hl_rel(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        """Hl rel for BatchedLongShortEnv.

        Args:
            high (torch.Tensor): The high value.
            low (torch.Tensor): The low value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        return 2.0 * (high - low) / (high + low + self.eps)

    def _time_features(self) -> torch.Tensor:
        """Time features for BatchedLongShortEnv.

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
        return torch.stack(features)

    def _obs(self) -> torch.Tensor:
        """Obs for BatchedLongShortEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        B, N = self.B, self.N
        V = self.V.clamp(min=self.eps)

        t_vec   = self._time_features()                             # (B, 6)
        x_cash  = (self.C / V)[:, None]                             # (B, 1)
        x_assets = (self.pos_units * self.p) / V[:, None]           # (B, N)  signed
        x_rel   = torch.cat([x_cash, x_assets], dim=1)             # (B, N+1)

        S     = self.committed.sum(dim=1)                           # (B,)
        c_rel = torch.where(
            S[:, None] > 0,
            self.committed / (S[:, None] + self.eps),
            torch.zeros(B, N, dtype=self.dtype),
        )
        rho   = S / (S + self.C + self.eps)

        has      = self.committed > self.eps
        pnl      = self.pos_units * (self.p - self.entry_price)     # (B, N) signed
        tax      = pnl.clamp(min=0.0) * self.tax_rate
        net      = pnl - tax
        unrl_rel = torch.where(
            has,
            net / self.committed.clamp(min=self.eps),
            torch.zeros(B, N, dtype=self.dtype),
        )

        side_rel = torch.zeros(B, N, dtype=self.dtype)
        side_rel[self.pos_units > self.eps]  = 1.0
        side_rel[self.pos_units < -self.eps] = -1.0

        return torch.cat([
            t_vec,
            self.p_rel.view(B, -1),
            self.hl_rel,
            x_rel,
            c_rel,
            rho[:, None],
            self.v_rel.view(B, -1),
            unrl_rel,
            side_rel,
            self.vol_rel.view(B, -1),
        ], dim=1)

    def _update(
        self,
        close: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        volume: torch.Tensor,
        time: torch.Tensor,
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

        dt = (t_new - self.t).clamp(min=0.0).to(self.dtype)
        self.t = t_new

        self.p = close

        alpha_p = 1.0 - torch.exp(
            -dt[:, None, None] / self.tau_p[None, :, None].clamp(min=self.eps)
        )

        p_exp = self.p[:, None, :]
        self.p_smooth = self.p_smooth + alpha_p * (p_exp - self.p_smooth)
        self.p_rel    = (p_exp - self.p_smooth) / self.p_smooth.clamp(min=self.eps)

        hl_new  = self._hl_rel(high, low)
        self.hl_rel = hl_new
        hl_exp  = hl_new[:, None, :]
        self.hl_smooth = self.hl_smooth + alpha_p * (hl_exp - self.hl_smooth)
        self.vol_rel   = (hl_exp - self.hl_smooth) / self.hl_smooth.clamp(min=self.eps)

        self.v = volume * self.p if self.use_dollar_volume else volume
        v_exp  = self.v[:, None, :]
        self.v_smooth = self.v_smooth + alpha_p * (v_exp - self.v_smooth)
        self.v_rel    = (v_exp - self.v_smooth) / self.v_smooth.clamp(min=self.eps)

    def _trade(self, actions: torch.Tensor) -> None:
        """Two-phase vectorized execution: close (incl. flip-triggered) then open.

        Args:
            actions (torch.Tensor): The actions value.

        Returns:
            None: This function does not return a value.
        """
        B, N = self.B, self.N
        b = self._b_idx

        self.realized_cost = torch.zeros(B, N, dtype=self.dtype)
        self.realized_pnl  = torch.zeros(B, N, dtype=self.dtype)

        # Action decode
        sell_offset  = 1 + N
        close_offset = 1 + 2 * N

        is_long_act  = (actions >= 1)             & (actions <  sell_offset)
        is_short_act = (actions >= sell_offset)   & (actions <  close_offset)
        is_close_act = (actions >= close_offset)  & (actions <  self.action_dim)

        long_asset  = (actions - 1).clamp(0, N - 1)
        short_asset = (actions - sell_offset).clamp(0, N - 1)
        close_asset = (actions - close_offset).clamp(0, N - 1)

        # Determine which asset (if any) to close this step:
        # - explicit close actions
        # - long action while currently short that asset (flip)
        # - short action while currently long that asset (flip)
        cur_long_on_long_act  = self.pos_units[b, long_asset]  > self.eps
        cur_short_on_long_act = self.pos_units[b, long_asset]  < -self.eps
        cur_long_on_short_act = self.pos_units[b, short_asset] > self.eps
        cur_short_on_short_a  = self.pos_units[b, short_asset] < -self.eps

        flip_long  = is_long_act  & cur_short_on_long_act   # close the short first
        flip_short = is_short_act & cur_long_on_short_act   # close the long first

        do_close = is_close_act | flip_long | flip_short

        # Pick which asset to close per env
        # Priority: explicit close -> flip-long -> flip-short (mutually exclusive in practice)
        target_close_asset = torch.where(
            is_close_act, close_asset,
            torch.where(flip_long, long_asset, short_asset),
        )

        # Compute close math at target_close_asset
        units_c     = self.pos_units[b, target_close_asset]
        entry_c     = self.entry_price[b, target_close_asset]
        committed_c = self.committed[b, target_close_asset]
        price_c     = self.p[b, target_close_asset]

        notional_c = torch.abs(units_c) * price_c
        pnl_c      = units_c * (price_c - entry_c)
        gross_c    = committed_c + pnl_c

        close_valid = do_close & (torch.abs(units_c) > self.eps) & (notional_c > self.c_fee)

        proceeds    = (gross_c - self.c_fee).clamp(min=0.0)
        pre_tax_pnl = proceeds - committed_c
        tax         = pre_tax_pnl.clamp(min=0.0) * self.tax_rate
        net_proceeds = proceeds - tax
        realized_pnl = net_proceeds - committed_c

        cv = close_valid.to(self.dtype)
        self.C = self.C + net_proceeds * cv

        # Zero the closed slot via scatter_add of the negative current value
        self.pos_units.scatter_add_(1, target_close_asset[:, None], -(units_c * cv)[:, None])
        self.committed.scatter_add_(1, target_close_asset[:, None], -(committed_c * cv)[:, None])
        self.entry_price.scatter_add_(1, target_close_asset[:, None], -(entry_c * cv)[:, None])

        self.realized_cost.scatter_add_(1, target_close_asset[:, None], (committed_c * cv)[:, None])
        self.realized_pnl.scatter_add_(1, target_close_asset[:, None], (realized_pnl * cv)[:, None])

        # If a flip tried to close and failed, the subsequent open must not fire.
        flip_failed_long  = flip_long  & ~close_valid
        flip_failed_short = flip_short & ~close_valid

        # ---- OPEN phase ----
        is_open = (is_long_act & ~flip_failed_long) | (is_short_act & ~flip_failed_short)
        open_asset = torch.where(is_long_act, long_asset, short_asset)
        open_side  = torch.where(is_long_act, torch.ones(B, dtype=self.dtype),
                                              -torch.ones(B, dtype=self.dtype))

        # Only open if target slot is now flat (safety — prevents double-open)
        cur_units_open = self.pos_units[b, open_asset]
        slot_flat = torch.abs(cur_units_open) <= self.eps

        buy_dollars = self.C - self.o_fee
        open_valid = is_open & slot_flat & (self.C > self.o_fee) & (buy_dollars >= self.min_open_dollars)

        open_price = self.p[b, open_asset].clamp(min=self.eps)
        units_mag  = buy_dollars / open_price
        open_units = open_side * units_mag
        total_cost = torch.minimum(buy_dollars + self.o_fee, self.C)

        ov = open_valid.to(self.dtype)
        self.C = self.C - total_cost * ov
        self.pos_units.scatter_add_(1, open_asset[:, None], (open_units * ov)[:, None])
        self.committed.scatter_add_(1, open_asset[:, None], (buy_dollars * ov)[:, None])
        self.entry_price.scatter_add_(1, open_asset[:, None], (open_price * ov)[:, None])

        # Dust cleanup: any slot with negligible committed is fully reset.
        dust = self.committed < self.transaction_eps
        self.pos_units   = self.pos_units.masked_fill(dust, 0.0)
        self.committed   = self.committed.masked_fill(dust, 0.0)
        self.entry_price = self.entry_price.masked_fill(dust, 0.0)
        self.C           = self.C.clamp(min=0.0)

    def _reward(self) -> torch.Tensor:
        """Reward for BatchedLongShortEnv.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        has_realized = self.realized_cost.sum(dim=1) > self.eps
        roi = self.realized_pnl.sum(dim=1) / self.realized_cost.sum(dim=1).clamp(min=self.eps)
        V      = self.V
        V_prev = self.V_prev.clamp(min=self.eps)
        if self.reward_mode == "return":
            baseline = self.val_coeff * (V - V_prev) / V_prev
        else:
            baseline = self.val_coeff * torch.log(V.clamp(min=self.eps) / V_prev)
        return torch.where(has_realized, self.roi_coeff * roi, baseline)

    # -------------------------------------------------------------------------
    # public step / mask
    # -------------------------------------------------------------------------

    def step(
        self,
        actions: torch.Tensor,
        close: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        volume: torch.Tensor,
        time: torch.Tensor,
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
        actions = torch.as_tensor(actions, dtype=torch.long).to(self.C.device).reshape(-1)
        if actions.numel() != self.B:
            raise ValueError(f"actions must have {self.B} elements, got {actions.numel()}")
        if bool(((actions < 0) | (actions >= self.action_dim)).any().item()):
            raise ValueError(f"Invalid discrete action, expected values in [0, {self.action_dim - 1}]")

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
        eps = self.eps

        mask = torch.zeros(B, self.action_dim, dtype=torch.bool)
        mask[:, 0] = True

        is_long  = self.pos_units > eps                    # (B, N)
        is_short = self.pos_units < -eps
        is_flat  = ~(is_long | is_short)

        notional = torch.abs(self.pos_units) * self.p
        close_ok = (~is_flat) & (notional > self.c_fee)    # (B, N)

        # cash-if-close for each asset independently
        pnl      = self.pos_units * (self.p - self.entry_price)
        gross    = self.committed + pnl
        proceeds = (gross - self.c_fee).clamp(min=0.0)
        pre_tax  = proceeds - self.committed
        tax      = pre_tax.clamp(min=0.0) * self.tax_rate
        net_proc = proceeds - tax                           # (B, N)
        cash_if_close = torch.where(
            close_ok,
            self.C[:, None] + net_proc,
            self.C[:, None].expand(B, N),
        )
        cash_avail = torch.where(is_flat, self.C[:, None].expand(B, N), cash_if_close)

        can_open = (cash_avail > self.o_fee) & ((cash_avail - self.o_fee) >= self.min_open_dollars)

        # long k: invalid if already long; otherwise need cash_avail + (close_ok if short)
        long_valid  = (~is_long)  & can_open & (is_flat | (is_short & close_ok))
        short_valid = (~is_short) & can_open & (is_flat | (is_long  & close_ok))

        mask[:, 1:1 + N]             = long_valid
        mask[:, 1 + N:1 + 2 * N]     = short_valid
        mask[:, 1 + 2 * N:1 + 3 * N] = close_ok

        return mask
