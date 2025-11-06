from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch


@dataclass
class State:
    """
    Observation returned by the environment.

    Attributes
    ----------
    p_rel : torch.Tensor
        Relative price deviations across time-constants, shape [M_p, N],
        with p_rel[m, i] = (p[i] - s[m, i]) / s[m, i], where s is the EWMA smoother.
    x : torch.Tensor
        Portfolio value weights, shape [N+1]:
        x[0] = C / V (cash fraction), x[i>0] = (w[i-1] * p[i-1]) / V.
    sigma : Optional[torch.Tensor]
        Optional volatility features (per asset, per tau), shape [M_v, N].
        sigma[m, i] = sqrt(EWMA(r_i^2; tau_v[m]) + eps). None if disabled.
    v_rel : Optional[torch.Tensor]
        Optional relative (dollar) volume features, shape [M_u, N]:
        v_rel[m, i] = (DV[i] - EWMA(DV; tau_u[m])) / EWMA(DV; tau_u[m]).
        None if disabled.
    """
    p_rel: torch.Tensor
    x: torch.Tensor
    sigma: Optional[torch.Tensor] = None
    v_rel: Optional[torch.Tensor] = None

    def to_tensor(self):
        """Concatenate all features into a single 1D tensor."""
        parts = [self.p_rel.flatten(), self.x.flatten()]
        if self.sigma is not None:
            parts.append(self.sigma.flatten())
        if self.v_rel is not None:
            parts.append(self.v_rel.flatten())
        return torch.cat(parts)


class MultiCurrencyEnv:
    r"""
    A lightweight "gym-like" multi-currency trading environment (no gym dependency).

    State (observation)
    -------------------
    The agent observes scale-free features only (no raw prices):
      • p_rel ∈ ℝ^{M_p×N}:  p_rel = (p − s) / s, where s is an EWMA smoother at multiple time-constants.
      • x ∈ ℝ^{N+1}:       value weights [cash fraction, per-asset exposure fractions].
      • sigma ∈ ℝ^{M_v×N} (optional): EWMA volatility (√EWMA(r^2)).
      • v_rel ∈ ℝ^{M_u×N} (optional): relative (dollar) volume vs. EWMA baseline.

    Actions
    -------
    a ∈ (-1,1)^{N+1}:
      • a[0] controls the fraction of *capital* C to allocate to buys in this step (mapped to [0,1)).
      • a[1:] per-asset: negative values => sell fraction of units; positive => eligibility
        for receiving a share of the buy budget (proportional among positives).

    Trading Mechanics (unit-accurate, with fees)
    --------------------------------------------
      Sell (for assets with a_k < 0):
        units_sold = (-a_k) * w_k, notional = units_sold * p_k
        fees_s = s_k * notional, proceeds = notional − fees_s
        C += sum(proceeds), w_k -= units_sold

      Buy (for assets with a_k > 0):
        budget I = bar_a0 * C with bar_a0 = (a0+1)/2 ∈ [0,1)
        split I across positives in proportion to their a_k
        per-asset spend A_k = share * I; fees_b = b_k * A_k
        units_bought = (A_k − fees_b) / p_k
        w_k += units_bought; C -= I

    Reward
    ------
      • "diff": r_t = V_{t+1} − V_t           (USD PnL)
      • "log" : r_t = log((V_{t+1}+eps)/(V_t+eps))  (continuously-compounded return)

    Termination
    -----------
      done if step_count ≥ max_steps (if provided) or V ≤ bankruptcy_threshold.

    Notes
    -----
      • Smoothers use forward-Euler: s ← s + (dt/τ) * (p − s)
      • Volatility uses EWMA of r^2,  r = log p_t − log p_{t−1}
      • Relative volume uses dollar-volume DV = volume * price by default.
    """

    # ---------- construction ----------

    def __init__(
        self,
        C0: float,
        w0: torch.Tensor,
        p0: torch.Tensor,
        sell_fee: torch.Tensor,
        buy_fee: torch.Tensor,
        taus_price: torch.Tensor,
        *,
        dt: float = 1.0,
        max_steps: Optional[int] = None,
        bankruptcy_threshold: float = 1.0,
        transaction_eps: float = 0.05,
        reward_mode: str = "log",
        # optional features
        vol_taus: Optional[torch.Tensor] = None,
        volu_taus: Optional[torch.Tensor] = None,
        use_dollar_volume: bool = True,
        # numerics / device
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-12,
        mode: str = "custom",
    ):
        """
        Parameters
        ----------
        C0 : float
            Initial capital (USD).
        w0 : torch.Tensor, shape [N]
            Initial units per asset.
        p0 : torch.Tensor, shape [N]
            Initial prices (USD/unit).
        sell_fee : torch.Tensor, shape [N]
            Fractional sell fees (per asset), in [0,1].
        buy_fee : torch.Tensor, shape [N]
            Fractional buy fees (per asset), in [0,1].
        taus_price : torch.Tensor, shape [M_p]
            Time-constants for the price EWMAs that define p_rel.
        dt : float, optional
            Simulation time-step used in Euler updates for all EWMAs.
        max_steps : Optional[int], optional
            Episode length cap. If None, episodes end only by bankruptcy check.
        bankruptcy_threshold : float, optional
            Terminates episode if V ≤ threshold.
        reward_mode : {"diff","log"}, optional
            Reward per step: USD PnL ("diff") or log-return ("log").
        vol_taus : Optional[torch.Tensor], shape [M_v], optional
            If provided, enables volatility features (EWMA variance, then sqrt).
        volu_taus : Optional[torch.Tensor], shape [M_u], optional
            If provided, enables relative (dollar) volume features.
        use_dollar_volume : bool, optional
            If True, volume inputs are multiplied by price to form dollar-volume.
        device : Optional[torch.device], optional
            Torch device (defaults to CPU).
        dtype : torch.dtype, optional
            Tensor dtype for internal state (defaults to float32).
        eps : float, optional
            Small constant for numerical safety (divisions/logs).
        """
        # shape checks
        assert w0.ndim == 1 and p0.ndim == 1
        assert w0.shape == p0.shape == sell_fee.shape == buy_fee.shape
        assert taus_price.ndim == 1 and taus_price.numel() > 0
        assert reward_mode in ("diff", "log")
        assert bankruptcy_threshold >= 0.0
        assert transaction_eps >= 0.0

        self.N = w0.shape[0]
        self.Mp = taus_price.numel()

        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.eps = float(eps)
        self.mode = mode

        # persistent parameters / buffers
        self.dt = float(dt)
        self.max_steps = max_steps
        self.bankruptcy_threshold = float(bankruptcy_threshold)
        self.reward_mode = reward_mode
        self.use_dollar_volume = bool(use_dollar_volume)
        self.transaction_eps = float(transaction_eps)

        self.s_fee = sell_fee.to(self.device, self.dtype).clamp(0.0, 1.0)  # [N]
        self.b_fee = buy_fee.to(self.device, self.dtype).clamp(0.0, 1.0)   # [N]

        self.taus_p = taus_price.to(self.device, self.dtype).clamp(min=1e-12)  # [M_p]
        self.vol_taus = (
            None if vol_taus is None
            else vol_taus.to(self.device, self.dtype).clamp(min=1e-12)
        )  # [M_v] or None
        self.volu_taus = (
            None if volu_taus is None
            else volu_taus.to(self.device, self.dtype).clamp(min=1e-12)
        )  # [M_u] or None

        # store defaults for resets
        self._defaults = dict(
            C0=float(C0),
            w0=w0.detach().clone(),
            p0=p0.detach().clone(),
        )

        # allocate state by calling reset()
        self.reset()

    @property
    def v(self) -> torch.Tensor:
        """Current per-asset exposures v_i = w_i * p_i (shape [N])."""
        return self.w * self.p

    @property
    def V(self) -> torch.Tensor:
        """Current portfolio value V = C + sum_i w_i * p_i (scalar)."""
        return self.C + torch.sum(self.v)
    
    @property
    def action_size(self) -> int:
        """
        Number of continuous action dimensions.

        Returns
        -------
        int
            Action dimensionality = N + 1.
            a[0]: capital allocation fraction,
            a[1:]: per-asset buy/sell signals.
        """
        return self.N + 1

    @property
    def state_size(self) -> int:
        """
        Flattened dimensionality of the observation vector.

        Components
        -----------
        - p_rel : (M_p * N)
        - x     : (N + 1)
        - sigma : (M_v * N)  [if enabled]
        - v_rel : (M_u * N)  [if enabled]

        Returns
        -------
        int
            Total flattened length of state features.
        """
        Mv = 0 if self.vol_taus is None else self.vol_taus.numel()
        Mu = 0 if self.volu_taus is None else self.volu_taus.numel()
        
        return self.Mp * self.N + (self.N + 1) + Mv * self.N + Mu * self.N

    def reset(
        self,
        C0: Optional[float] = None,
        w0: Optional[torch.Tensor] = None,
        p0: Optional[torch.Tensor] = None,
        volume0: Optional[torch.Tensor] = None,
        sell_fee: Optional[torch.Tensor] = None,
        buy_fee: Optional[torch.Tensor] = None,
    ) -> State:
        """
        Reset the environment to an initial state.

        Assumptions
        -----------
        - Market data will be provided **externally** after reset by setting:
            self.p       # prices [N], torch.float, on env device/dtype
            self.volume  # volumes [N], optional; if enabled and None, v_rel stays zeros
        The `update()` method reads only `self.p` and `self.volume`.

        Parameters
        ----------
        C0, w0, p0, volume0 : optional
            Optional overrides for initial capital, wallets, prices, and volume.
            If `volume0` is None and volume features are enabled, an internal
            tiny baseline is used so shapes are stable and v_rel starts at zeros.
        sell_fee, buy_fee : Optional[torch.Tensor], shape [N]
            Optional fee overrides applied at reset.

        Returns
        -------
        State
            Initial observation (p_rel, x[, sigma][, v_rel]).
            At reset, p_rel == 0, sigma ~ sqrt(eps), v_rel == 0 (if enabled).
        """
        # Resolve initial scalars/vectors
        C0 = float(self._defaults["C0"] if C0 is None else C0)
        w0 = self._defaults["w0"] if w0 is None else w0
        p0 = self._defaults["p0"] if p0 is None else p0

        if sell_fee is not None:
            self.s_fee = sell_fee.to(self.device, self.dtype).clamp(0.0, 1.0)
        if buy_fee is not None:
            self.b_fee = buy_fee.to(self.device, self.dtype).clamp(0.0, 1.0)

        # Core account state
        self.C = torch.tensor(C0, dtype=self.dtype, device=self.device)
        self.w = w0.to(self.device, self.dtype).clone()

        # Prices & Volume are part of the env state and may be updated externally later.
        self.p = p0.to(self.device, self.dtype).clone()                          # [N]
        self.volume = None if volume0 is None else volume0.to(self.device, self.dtype)  # [N] or None

        # Price smoothers (start at current price so p_rel starts at zero)
        self.p_smooth = self.p[None, :].repeat(self.Mp, 1).clone()               # [M_p, N]
        self.p_prev = self.p.clone()                                             # for returns

        # Reward bookkeeping
        self._prev_V = self.V.detach()
        self.step_count = 0

        # Volatility trackers (stable shape even before first update)
        if self.vol_taus is not None:
            Mv = self.vol_taus.numel()
            self.q_var = torch.zeros((Mv, self.N), dtype=self.dtype, device=self.device)  # EWMA var
            self.sigma = torch.sqrt(self.q_var + self.eps)                                 # ~0 initially
        else:
            self.q_var = None
            self.sigma = None

        # Volume trackers (stable shape even if volume is not provided yet)
        if self.volu_taus is not None:
            Mu = self.volu_taus.numel()
            if self.volume is None:
                # tiny positive baseline to avoid div-by-zero; v_rel starts at zeros
                self.v_smooth = torch.full((Mu, self.N), 1e-6, dtype=self.dtype, device=self.device)
            else:
                v0 = self.volume if not self.use_dollar_volume else self.volume * self.p
                self.v_smooth = v0[None, :].repeat(Mu, 1).clone()
            self.v_rel = torch.zeros_like(self.v_smooth)                         # [M_u, N]
        else:
            self.v_smooth = None
            self.v_rel = None

        # Initial observation: p_rel = 0, x from current holdings, sigma/v_rel as above
        return self._get_state()

    
    def update_prices(self, p: torch.Tensor):
        """
        Update the environment with new price data for the current step.

        Parameters
        ----------
        p : torch.Tensor, shape [N]
            New price data for each asset.
        """
        self.p = p.to(self.device, self.dtype)

    def update_volume(self, volume: torch.Tensor):
        """
        Update the environment with new volume data for the current step.

        Parameters
        ----------
        volume : torch.Tensor, shape [N]
            New volume data for each asset.
        """
        self.volume = volume.to(self.device, self.dtype)
        self._last_v = self.volume.clone()

    def _update(self) -> None:
        """
        Update all internal smoothers and derived features based on
        the current self.p (prices) and self.volume (volumes).

        This method assumes self.p and self.volume are set externally
        (for example, by a live API or a data feed handler).
        """
        assert hasattr(self, "p"), "self.p (prices) must be set externally"
        assert self.p is not None, "self.p must not be None"

        p = self.p.to(self.device, self.dtype)

        # ----- PRICE SMOOTHERS -----
        factor_p = (self.dt / self.taus_p)[:, None]             # [M_p,1]
        self.p_smooth += factor_p * (p[None, :] - self.p_smooth)

        # ----- VOLATILITY -----
        if self.vol_taus is not None:
            r = (p.clamp(min=self.eps).log() - self.p_prev.clamp(min=self.eps).log())
            r2 = r * r
            factor_v = (self.dt / self.vol_taus)[:, None]
            if self.q_var is None:
                self.q_var = r2[None, :].repeat(self.vol_taus.numel(), 1)
            else:
                self.q_var += factor_v * (r2[None, :] - self.q_var)
            self.sigma = torch.sqrt(self.q_var.clamp(min=0.0) + self.eps)
            self.p_prev = p.clone()
        else:
            self.sigma = None

        # ----- VOLUME -----
        if self.volu_taus is not None:
            assert hasattr(self, "volume"), "self.volume must be set externally"
            v = self.volume.to(self.device, self.dtype)
            if self.use_dollar_volume:
                v = v * p
            factor_u = (self.dt / self.volu_taus)[:, None]
            if self.v_smooth is None:
                self.v_smooth = v[None, :].repeat(self.volu_taus.numel(), 1)
            else:
                self.v_smooth += factor_u * (v[None, :] - self.v_smooth)
            self.v_rel = (v[None, :] - self.v_smooth) / self.v_smooth.clamp(min=self.eps)
        else:
            self.v_rel = None

    def step(self, a: torch.Tensor) -> Tuple[State, float, bool, Dict]:
        """
        Advance one environment step: execute trades, update marks/smoothers/features.

        Parameters
        ----------
        a : torch.Tensor, shape [N+1]
            Action vector in (-1,1)^{N+1}. a[0] controls buy budget fraction; a[1:]
            negative => sell fraction; positive => participates in buy allocation.

        Returns
        -------
        state : State
            Next observation (p_rel, x[, sigma][, v_rel]).
        reward : float
            Per-step reward ("diff" PnL or "log" log-return).
        done : bool
            Whether the episode has terminated.
        info : Dict
            Diagnostics (numpy copies of masks, fractions, and features).
        """
        assert a.shape[-1] == self.N + 1, "Action must have N+1 outputs"
        a = a.to(self.device, self.dtype)

        a0 = a[0]
        a_cur = a[1:]

        # ----- SELLING -----
        sell_mask = a_cur < -self.transaction_eps
        f_sell = torch.zeros_like(a_cur)
        f_sell[sell_mask] = (-a_cur[sell_mask]).clamp(0.0, 1.0)
        if torch.any(sell_mask):
            units_sold = f_sell * self.w
            notional = units_sold * self.p
            fees_s = self.s_fee * notional
            proceeds = notional - fees_s
            self.C = self.C + proceeds.sum()
            self.w = self.w - units_sold

        # ----- BUYING -----
        buy_mask = a_cur > self.transaction_eps
        if torch.any(buy_mask):
            positives = a_cur[buy_mask]
            denom = positives.sum()
            if denom.item() > 1e-12:
                weights = torch.zeros_like(a_cur)
                weights[buy_mask] = positives / denom  # normalized among positives

                # budget fraction in [0,1)
                a0_bar = 0.5 * (a0 + 1.0)
                a0_bar = a0_bar.clamp(0.0, 1.0)

                I = a0_bar * self.C
                if I.item() > 0.0:
                    A = weights * I                    # USD per asset pre-fee
                    fee_b = self.b_fee * A
                    usd_to_units = (A - fee_b) / self.p.clamp(min=1e-12)
                    self.w = self.w + usd_to_units
                    self.C = self.C - I

        # ----- MARK-TO-MARKET -----
        self._update()

        # ----- REWARD -----
        V_now = self.V
        if self.reward_mode == "diff":
            reward = float((V_now - self._prev_V).detach().cpu())
        else:
            v_prev = float(self._prev_V.detach().cpu())
            v_now = float(V_now.detach().cpu())
            reward = float(torch.log(torch.tensor((v_now + self.eps) / (v_prev + self.eps))).item())
        self._prev_V = V_now.detach()

        # ----- TERMINATION -----
        self.step_count += 1
        done = (self.max_steps is not None and self.step_count >= self.max_steps) \
            or (float(V_now.detach().cpu()) <= self.bankruptcy_threshold)

        # ----- INFO DICT -----
        info = {
            "a": a.detach().cpu().numpy(),
            "sell_mask": sell_mask.detach().cpu().numpy(),
            "buy_mask": buy_mask.detach().cpu().numpy(),
            "f_sell": f_sell.detach().cpu().numpy(),
            "p": self.p.detach().cpu().numpy(),
            "w": self.w.detach().cpu().numpy(),
            "C": float(self.C.detach().cpu()),
            "V": float(V_now.detach().cpu()),
            "step": int(self.step_count),
        }

        return self._get_state(), reward, done, info

    def _compute_value_weights(self) -> torch.Tensor:
        """
        Compute value weights (cash + per-asset exposures) normalized by portfolio value.

        Returns
        -------
        torch.Tensor, shape [N+1]
            x[0] = C/V, x[i>0] = (w[i-1]*p[i-1])/V. Guaranteed to sum to 1 within fp error.
        """
        V = self.V.clamp(min=1e-12)
        
        x_cash = (self.C / V)[None]          # [1]
        x_assets = self.v / V     # [N]
        
        return torch.cat([x_cash, x_assets], dim=0)

    def _get_state(self) -> State:
        """
        Assemble the current observation from internals (after the latest `update()`).

        Returns
        -------
        State
            (p_rel, x[, sigma][, v_rel]) where
            - p_rel[m, i] = (p[i] - p_smooth[m, i]) / p_smooth[m, i]
            - x = [ C/V , (w * p) / V ]
            - sigma, v_rel present iff their τ-grids are enabled.
        """
        # price-relative features across taus
        p_rel = (self.p[None, :] - self.p_smooth) / self.p_smooth.clamp(min=self.eps)  # [M_p, N]

        # value weights
        x = self._compute_value_weights()                                              # [N+1]

        # optional blocks (kept stable by `update()`)
        sigma = self.sigma if self.vol_taus is not None else None
        v_rel = self.v_rel if self.volu_taus is not None else None

        return State(
            p_rel=p_rel.clone(),
            x=x.clone(),
            sigma=None if sigma is None else sigma.clone(),
            v_rel=None if v_rel is None else v_rel.clone(),
        )
