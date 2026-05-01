from typing import Tuple, Dict

import torch

from src.environment.generic.longshort import (
    BatchedMultiCurrencyEnv,
    MultiCurrencyEnv,
    State,
    StateHistory,
)


def _split_hier_action(a) -> tuple[int, int]:
    """Decode a single hierarchical action into (a_d, a_q) ints."""
    if a is None:
        return 0, 0
    if isinstance(a, (tuple, list)):
        a_d, a_q = a
    elif isinstance(a, torch.Tensor):
        flat = a.reshape(-1)
        if flat.numel() != 2:
            raise ValueError(f"Hierarchical action tensor must have 2 elements, got {flat.numel()}.")
        a_d, a_q = flat[0], flat[1]
    else:
        raise TypeError(f"Unsupported hierarchical action type: {type(a)}")

    a_d = int(a_d.item() if isinstance(a_d, torch.Tensor) else a_d)
    a_q = int(a_q.item() if isinstance(a_q, torch.Tensor) else a_q)
    return a_d, a_q


def _split_hier_actions_batched(actions, B: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a batched hierarchical action into (a_d (B,), a_q (B,)) long tensors."""
    if actions is None:
        a_d = torch.zeros(B, dtype=torch.long, device=device)
        a_q = torch.zeros(B, dtype=torch.long, device=device)
        return a_d, a_q
    if isinstance(actions, (tuple, list)):
        a_d, a_q = actions
        a_d = torch.as_tensor(a_d).to(device=device, dtype=torch.long).reshape(-1)
        a_q = torch.as_tensor(a_q).to(device=device, dtype=torch.long).reshape(-1)
    else:
        t = torch.as_tensor(actions).to(device=device, dtype=torch.long)
        if t.dim() == 1 and t.numel() == B:
            # Primary-only input: bucket defaults to 0 (harmless for hold/close).
            a_d = t
            a_q = torch.zeros(B, dtype=torch.long, device=device)
        elif t.dim() >= 2 and t.shape[-1] == 2:
            t = t.reshape(B, 2)
            a_d, a_q = t[:, 0], t[:, 1]
        else:
            raise ValueError(
                f"Batched hierarchical actions must have shape (B, 2) or (B,), got {tuple(t.shape)}."
            )
    if a_d.numel() != B or a_q.numel() != B:
        raise ValueError(f"Hierarchical actions must each have {B} elements, got ({a_d.numel()}, {a_q.numel()}).")
    return a_d, a_q


class LongShortHierarchicalEnv(MultiCurrencyEnv):
    """
    Hierarchical bucketized long/short trading environment.

    Hybrid action: (a_d, a_q)
        a_d in [0, 3N]:
            0                          -> hold
            1   .. N                   -> BUY    asset i = a_d - 1     (open long)
            N+1 .. 2N                  -> SELL   asset i = a_d - 1 - N (open short)
            2N+1 .. 3N                 -> CLOSE  asset i = a_d - 1 - 2N (full close)
        a_q in [0, K-1]:
            bucket index for the size head; consumed only when buying or
            selling. Hold and close ignore a_q.

    The actor network is expected to emit ``1 + 3*N + 2*K`` logits, split as:
        [primary (1+3N) | buy_bucket (K) | sell_bucket (K)]

    The mask returned by ``valid_action_mask()`` is a dict with three tensors:
        primary : (1+3N,) bool          per-direction validity
        buy     : (N, K)  bool          per-asset, per-bucket buy validity
        sell    : (N, K)  bool          per-asset, per-bucket sell validity
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
        size_buckets: Tuple[float, ...] = (0.50, 1.00),
        done_reward_penalty: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
        eps: float = 1e-8,
    ):
        super().__init__(
            N=N,
            C0=C0,
            tau_p=tau_p,
            bankruptcy_threshold=bankruptcy_threshold,
            transaction_eps=transaction_eps,
            use_dollar_volume=use_dollar_volume,
            save_history=save_history,
            open_fee=open_fee,
            close_fee=close_fee,
            tax_rate=tax_rate,
            min_open_dollars=min_open_dollars,
            val_coeff=val_coeff,
            roi_coeff=roi_coeff,
            reward_mode=reward_mode,
            done_reward_penalty=done_reward_penalty,
            dtype=dtype,
            device=device,
            eps=eps,
        )

        assert len(size_buckets) > 0, "size_buckets must not be empty"
        assert all(0.0 < x <= 1.0 for x in size_buckets), "all size buckets must be in (0, 1]"

        self.size_buckets = torch.tensor(size_buckets, dtype=self.dtype, device=self.tau_p.device)
        self.K = int(self.size_buckets.numel())
        self.primary_action_dim = 1 + 3 * self.N
        self.action_dim = self.primary_action_dim + 2 * self.K

    # ------------------------------------------------------------------
    # mask
    # ------------------------------------------------------------------

    def valid_action_mask(self) -> Dict[str, torch.Tensor]:
        device = self._state_device()
        N, K = self.N, self.K

        primary = torch.zeros(self.primary_action_dim, dtype=torch.bool, device=device)
        buy_mask = torch.zeros(N, K, dtype=torch.bool, device=device)
        sell_mask = torch.zeros(N, K, dtype=torch.bool, device=device)

        primary[0] = True
        cash = float(self.C.item())

        for k in range(N):
            units = float(self.pos_units[k].item())
            is_long = units > self.eps
            is_short = units < -self.eps
            is_flat = not (is_long or is_short)

            price = float(self.p[k].item())
            notional_full = abs(units) * price
            close_ok_full = (not is_flat) and notional_full > self.c_fee

            if is_flat:
                cash_avail = cash
            elif close_ok_full:
                committed_k = float(self.committed[k].item())
                entry = float(self.entry_price[k].item())
                pnl = units * (price - entry)
                gross = committed_k + pnl
                proceeds = max(gross - self.c_fee, 0.0)
                pre_tax = proceeds - committed_k
                tax = max(pre_tax, 0.0) * self.tax_rate
                net_proceeds = proceeds - tax
                cash_avail = cash + net_proceeds
            else:
                cash_avail = cash

            any_buy = False
            any_sell = False
            for b_idx in range(K):
                frac = float(self.size_buckets[b_idx].item())
                budget = frac * cash_avail
                can_open = budget > self.o_fee and (budget - self.o_fee) >= self.min_open_dollars

                if is_long:
                    bv = False
                elif is_short:
                    bv = close_ok_full and can_open
                else:
                    bv = can_open
                buy_mask[k, b_idx] = bv
                any_buy = any_buy or bv

                if is_short:
                    sv = False
                elif is_long:
                    sv = close_ok_full and can_open
                else:
                    sv = can_open
                sell_mask[k, b_idx] = sv
                any_sell = any_sell or sv

            primary[1 + k] = any_buy
            primary[1 + N + k] = any_sell
            primary[1 + 2 * N + k] = close_ok_full

        return {"primary": primary, "buy": buy_mask, "sell": sell_mask}

    # ------------------------------------------------------------------
    # trading
    # ------------------------------------------------------------------

    def _trade(self, a) -> tuple[torch.Tensor, bool]:
        self.realized_cost.zero_()
        self.realized_pnl.zero_()

        a_d, a_q = _split_hier_action(a)

        if not (0 <= a_d < self.primary_action_dim):
            raise ValueError(f"Invalid primary action {a_d}, expected in [0, {self.primary_action_dim - 1}].")
        if not (0 <= a_q < self.K):
            raise ValueError(f"Invalid bucket action {a_q}, expected in [0, {self.K - 1}].")

        action_info = torch.tensor([0.0, -1.0, 0.0], dtype=self.dtype, device=self._state_device())

        if a_d == 0:
            return action_info, True

        N = self.N
        if 1 <= a_d <= N:
            kind = "long"
            k = a_d - 1
            frac = float(self.size_buckets[a_q].item())
        elif N + 1 <= a_d <= 2 * N:
            kind = "short"
            k = a_d - 1 - N
            frac = float(self.size_buckets[a_q].item())
        else:
            kind = "close"
            k = a_d - 1 - 2 * N
            frac = 1.0

        action_info[1] = float(k)
        action_info[2] = frac

        if kind == "close":
            action_info[0] = 3.0
            did = self._close_position(k, frac)
            return action_info, did

        side = 1 if kind == "long" else -1
        action_info[0] = 1.0 if side == 1 else 2.0

        units = float(self.pos_units[k].item())
        if (side == 1 and units < -self.eps) or (side == -1 and units > self.eps):
            self._close_position(k, frac=1.0)

        units_after = float(self.pos_units[k].item())
        if (side == 1 and units_after > self.eps) or (side == -1 and units_after < -self.eps):
            return action_info, False

        did = self._open_position(k, side, frac)

        dust = self.committed < self.transaction_eps
        self.pos_units[dust] = 0.0
        self.committed[dust] = 0.0
        self.entry_price[dust] = 0.0
        self.C = torch.clamp(self.C, min=0.0)

        return action_info, did


class BatchedLongShortHierarchicalEnv(BatchedMultiCurrencyEnv):
    """
    Fully batched hierarchical long/short environment.

    Action: tuple ``(a_d, a_q)`` of two ``(B,)`` long tensors, or a single
    ``(B, 2)`` long tensor with ``[..., 0] = a_d`` and ``[..., 1] = a_q``.

    Mask returned by ``valid_action_mask()``:
        primary : (B, 1+3N) bool
        buy     : (B, N, K) bool
        sell    : (B, N, K) bool
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
        size_buckets: Tuple[float, ...] = (0.50, 1.00),
        done_reward_penalty: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
        eps: float = 1e-8,
    ):
        super().__init__(
            B=B,
            N=N,
            C0=C0,
            tau_p=tau_p,
            bankruptcy_threshold=bankruptcy_threshold,
            transaction_eps=transaction_eps,
            use_dollar_volume=use_dollar_volume,
            open_fee=open_fee,
            close_fee=close_fee,
            tax_rate=tax_rate,
            min_open_dollars=min_open_dollars,
            val_coeff=val_coeff,
            roi_coeff=roi_coeff,
            reward_mode=reward_mode,
            done_reward_penalty=done_reward_penalty,
            dtype=dtype,
            device=device,
            eps=eps,
        )

        assert len(size_buckets) > 0, "size_buckets must not be empty"
        assert all(0.0 < x <= 1.0 for x in size_buckets), "all size buckets must be in (0, 1]"

        self.size_buckets = torch.tensor(size_buckets, dtype=self.dtype, device=self.tau_p.device)
        self.K = int(self.size_buckets.numel())
        self.primary_action_dim = 1 + 3 * self.N
        self.action_dim = self.primary_action_dim + 2 * self.K

    # ------------------------------------------------------------------
    # trading
    # ------------------------------------------------------------------

    def _trade(self, actions) -> None:
        device = self._state_device()
        a_d, a_q = _split_hier_actions_batched(actions, self.B, device, self.dtype)

        if bool(((a_d < 0) | (a_d >= self.primary_action_dim)).any().item()):
            raise ValueError(f"Invalid primary actions, expected in [0, {self.primary_action_dim - 1}].")
        if bool(((a_q < 0) | (a_q >= self.K)).any().item()):
            raise ValueError(f"Invalid bucket actions, expected in [0, {self.K - 1}].")

        N = self.N
        sell_offset = 1 + N
        close_offset = 1 + 2 * N

        is_buy_act = (a_d >= 1) & (a_d < sell_offset)
        is_sell_act = (a_d >= sell_offset) & (a_d < close_offset)
        is_close_act = (a_d >= close_offset) & (a_d < self.primary_action_dim)

        buckets = self.size_buckets.to(device=device)
        bucket_frac = buckets[a_q]

        # Translate to the parent base action layout: 0=hold, 1..N=long, N+1..2N=short, 2N+1..3N=close
        base_actions = torch.zeros(self.B, dtype=torch.long, device=device)
        base_actions = torch.where(is_buy_act, 1 + (a_d - 1).clamp(0, N - 1), base_actions)
        base_actions = torch.where(is_sell_act, 1 + N + (a_d - sell_offset).clamp(0, N - 1), base_actions)
        base_actions = torch.where(is_close_act, 1 + 2 * N + (a_d - close_offset).clamp(0, N - 1), base_actions)

        # Buy/sell use the bucket fraction; close uses full position (frac=1).
        frac = torch.where(
            is_close_act,
            torch.ones(self.B, dtype=self.dtype, device=device),
            torch.where(is_buy_act | is_sell_act, bucket_frac, torch.zeros(self.B, dtype=self.dtype, device=device)),
        )

        super()._trade((base_actions, frac))

    # ------------------------------------------------------------------
    # mask
    # ------------------------------------------------------------------

    def valid_action_mask(self) -> Dict[str, torch.Tensor]:
        B, N, K = self.B, self.N, self.K
        eps = self.eps
        device = self._state_device()

        is_long = self.pos_units > eps
        is_short = self.pos_units < -eps
        is_flat = ~(is_long | is_short)

        notional_full = torch.abs(self.pos_units) * self.p
        close_ok_full = (~is_flat) & (notional_full > self.c_fee)

        pnl = self.pos_units * (self.p - self.entry_price)
        gross = self.committed + pnl
        proceeds = (gross - self.c_fee).clamp(min=0.0)
        pre_tax = proceeds - self.committed
        tax = pre_tax.clamp(min=0.0) * self.tax_rate
        net_proc = proceeds - tax

        cash_if_close = torch.where(
            close_ok_full,
            self.C[:, None] + net_proc,
            self.C[:, None].expand(B, N),
        )
        cash_avail = torch.where(is_flat, self.C[:, None].expand(B, N), cash_if_close)  # (B, N)

        buckets = self.size_buckets.to(device=device)                                    # (K,)
        budget = cash_avail[:, :, None] * buckets[None, None, :]                         # (B, N, K)
        can_open = (budget > self.o_fee) & ((budget - self.o_fee) >= self.min_open_dollars)

        long_valid = (~is_long[:, :, None]) & can_open & (
            is_flat[:, :, None] | (is_short[:, :, None] & close_ok_full[:, :, None])
        )
        short_valid = (~is_short[:, :, None]) & can_open & (
            is_flat[:, :, None] | (is_long[:, :, None] & close_ok_full[:, :, None])
        )

        buy_mask = long_valid                # (B, N, K)
        sell_mask = short_valid              # (B, N, K)

        primary = torch.zeros(B, self.primary_action_dim, dtype=torch.bool, device=device)
        primary[:, 0] = True
        primary[:, 1:1 + N] = buy_mask.any(dim=-1)                  # (B, N)
        primary[:, 1 + N:1 + 2 * N] = sell_mask.any(dim=-1)         # (B, N)
        primary[:, 1 + 2 * N:1 + 3 * N] = close_ok_full             # (B, N)

        return {"primary": primary, "buy": buy_mask, "sell": sell_mask}


__all__ = [
    "State",
    "StateHistory",
    "LongShortHierarchicalEnv",
    "BatchedLongShortHierarchicalEnv",
]
