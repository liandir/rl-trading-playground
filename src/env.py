from dataclasses import dataclass
import torch


@dataclass
class TradeResult:
    C: torch.Tensor           # updated capital
    w: torch.Tensor           # updated wallets
    V: torch.Tensor           # portfolio value after trade
    info: dict                # diagnostics


class MultiCurrencyEnv:
    """
    State:
        - C: USD capital, shape []
        - w: units per currency, shape [N]
        - p: prices in USD per unit, shape [N]
        - s: sell fee fraction per unit notional, shape [N]
        - b: buy fee fraction per unit notional, shape [N]
    Actions:
        a in (-1,1)^{N+1}, with a[0] controlling buy budget fraction,
        a[1:] mapped to per-currency sell/buy signals.
    Mechanics use correct unit accounting:
        Sell fraction f_k of units w_k at price p_k
            proceeds = p_k * f_k * w_k
            fee_s = s_k * proceeds
            C += proceeds - fee_s
            w_k *= (1 - f_k)
        Buy allocation I = bar_a0 * C, split by positive signals
            A_l = bar_a_l * I  (USD)
            fee_b = b_l * A_l
            units_bought = (A_l - fee_b) / p_l
            w_l += units_bought
            C -= I  (fees are inside I)
    """
    def __init__(self, 
                 C0: float, 
                 w0: torch.Tensor, 
                 p0: torch.Tensor, 
                 sell_fee: torch.Tensor, 
                 buy_fee: torch.Tensor, 
                 device: torch.device | None = None):
        assert w0.ndim == 1 and p0.ndim == 1
        assert w0.shape == p0.shape == sell_fee.shape == buy_fee.shape
        self.N = w0.shape[0]
        self.device = device or torch.device("cpu")
        self.C = torch.tensor(C0, dtype=torch.float32, device=self.device)
        self.w = w0.to(self.device).float()
        self.p = p0.to(self.device).float()
        self.s = sell_fee.to(self.device).float().clamp(min=0.0, max=1.0)
        self.b = buy_fee.to(self.device).float().clamp(min=0.0, max=1.0)

    @property
    def V(self) -> torch.Tensor:
        return self.C + torch.sum(self.w * self.p)

    def step(self, a: torch.Tensor, new_prices: torch.device | None = None) -> TradeResult:
        r"""
        Apply trade given action vector a \in (-1,1)^{N+1}.
        Optionally pass new_prices to mark-to-market after trading.
        Returns updated state and diagnostics.
        """
        assert a.shape[-1] == self.N + 1, "Action must have N+1 outputs"
        a = a.to(self.device).float().clamp(-0.999, 0.999)  # avoid boundary pathologies

        # Parse action
        a0 = a[0]
        a_cur = a[1:]
        # Selling: indices with a_k < 0
        sell_mask = a_cur < 0
        # Fraction of units to sell in [0,1)
        f_sell = torch.zeros_like(a_cur)
        f_sell[sell_mask] = (-a_cur[sell_mask]).clamp(0.0, 1.0)

        # Execute sells
        if torch.any(sell_mask):
            units_sold = f_sell * self.w
            notional = units_sold * self.p
            fees_s = self.s * notional
            proceeds = notional - fees_s
            self.C = self.C + proceeds.sum()
            self.w = self.w - units_sold

        # Buying: indices with a_k > 0 and budget fraction a0
        buy_mask = a_cur > 0
        if torch.any(buy_mask):
            positives = a_cur[buy_mask]
            # normalized weights sum to 1
            denom = positives.sum()
            # numerical guard
            if denom.item() > 1e-12:
                bar_a = torch.zeros_like(a_cur)
                bar_a[buy_mask] = positives / denom
                # Budget fraction in [0,1)
                bar_a0 = 0.5 * (a0 + 1.0)  # map (-1,1)->(0,1)
                bar_a0 = bar_a0.clamp(0.0, 1.0)

                # Investable USD
                I = bar_a0 * self.C
                if I.item() > 0.0:
                    A = bar_a * I  # USD per asset pre-fee
                    fee_b = self.b * A
                    usd_to_units = (A - fee_b) / self.p
                    self.w = self.w + usd_to_units
                    # reduce capital by allocated I
                    self.C = self.C - I

        # Mark-to-market with new prices if given
        if new_prices is not None:
            assert new_prices.shape == self.p.shape
            self.p = new_prices.to(self.device).float()

        info = {
            "sell_mask": sell_mask.detach().cpu().numpy(),
            "buy_mask": buy_mask.detach().cpu().numpy(),
            "f_sell": f_sell.detach().cpu().numpy(),
            "C": float(self.C.detach().cpu()),
            "w": self.w.detach().cpu().numpy(),
            "p": self.p.detach().cpu().numpy(),
            "V": float(self.V.detach().cpu())
        }
        return TradeResult(self.C.clone(), self.w.clone(), self.V.clone(), info)

