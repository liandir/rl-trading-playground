from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class TickerSnapshot:
    bid: float
    ask: float
    last: float
    volume_today: float | None = None
    volume_24h: float | None = None
    timestamp: float | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(slots=True)
class BookSnapshot:
    best_bid_price: float | None = None
    best_bid_size: float | None = None
    best_ask_price: float | None = None
    best_ask_size: float | None = None
    spread: float | None = None
    mid: float | None = None


@dataclass(slots=True)
class TradeEvent:
    pair: str
    price: float
    volume: float
    timestamp: float
    side: str
    order_type: str


@dataclass(slots=True)
class PairSnapshot:
    pair: str
    timestamp: float
    ticker: TickerSnapshot | None = None
    book: BookSnapshot | None = None
    last_trade: TradeEvent | None = None
    trade_volume: float = 0.0
    trade_count: int = 0

    def copy(self) -> "PairSnapshot":
        return copy.deepcopy(self)

    @property
    def price(self) -> float | None:
        if self.last_trade is not None:
            return self.last_trade.price
        if self.ticker is not None:
            return self.ticker.last
        if self.book is not None:
            return self.book.mid
        return None


@dataclass(slots=True)
class MarketFrame:
    time: float
    prices: dict[str, float]
    volume: dict[str, float]
    source: str = "kraken"

    def to_env_input(
        self,
        pair_order: list[str] | tuple[str, ...],
        *,
        dtype: "torch.dtype | None" = None,
        time_dtype: "torch.dtype | None" = None,
    ) -> dict[str, "torch.Tensor"]:
        import torch

        dtype = dtype or torch.float32
        time_dtype = time_dtype or torch.float64

        return {
            "time": torch.tensor(float(self.time), dtype=time_dtype),
            "prices": torch.tensor([float(self.prices[pair]) for pair in pair_order], dtype=dtype),
            "volume": torch.tensor([float(self.volume.get(pair, 0.0)) for pair in pair_order], dtype=dtype),
        }
