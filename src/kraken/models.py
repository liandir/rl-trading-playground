"""Models utilities for Kraken market data and exchange integration helpers."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class TickerSnapshot:
    """TickerSnapshot market data snapshot for Kraken market data and exchange integration helpers."""
    bid: float
    ask: float
    last: float
    volume_today: float | None = None
    volume_24h: float | None = None
    timestamp: float | None = None

    @property
    def mid(self) -> float:
        """Mid for TickerSnapshot.

        Returns:
            float: The computed or requested result.
        """
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        """Spread for TickerSnapshot.

        Returns:
            float: The computed or requested result.
        """
        return self.ask - self.bid


@dataclass(slots=True)
class BookSnapshot:
    """BookSnapshot market data snapshot for Kraken market data and exchange integration helpers."""
    best_bid_price: float | None = None
    best_bid_size: float | None = None
    best_ask_price: float | None = None
    best_ask_size: float | None = None
    spread: float | None = None
    mid: float | None = None


@dataclass(slots=True)
class TradeEvent:
    """TradeEvent event record for Kraken market data and exchange integration helpers."""
    pair: str
    price: float
    volume: float
    timestamp: float
    side: str
    order_type: str


@dataclass(slots=True)
class PairSnapshot:
    """PairSnapshot market data snapshot for Kraken market data and exchange integration helpers."""
    pair: str
    timestamp: float
    ticker: TickerSnapshot | None = None
    book: BookSnapshot | None = None
    last_trade: TradeEvent | None = None
    trade_volume: float = 0.0
    trade_count: int = 0

    def copy(self) -> "PairSnapshot":
        """Return a copy of this object.

        Returns:
            'PairSnapshot': The computed or requested result.
        """
        return copy.deepcopy(self)

    @property
    def price(self) -> float | None:
        """Price for PairSnapshot.

        Returns:
            float | None: The computed or requested result.
        """
        if self.last_trade is not None:
            return self.last_trade.price
        if self.ticker is not None:
            return self.ticker.last
        if self.book is not None:
            return self.book.mid
        return None


@dataclass(slots=True)
class MarketFrame:
    """MarketFrame market frame for Kraken market data and exchange integration helpers."""
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
        """Convert the value to env input.

        Args:
            pair_order (list[str] | tuple[str, ...]): The pair order value.
            dtype ('torch.dtype | None'): The dtype value. Defaults to ``None``.
            time_dtype ('torch.dtype | None'): The time dtype value. Defaults to ``None``.

        Returns:
            dict[str, 'torch.Tensor']: The computed or requested result.
        """
        import torch

        dtype = dtype or torch.float32
        time_dtype = time_dtype or torch.float64

        return {
            "time": torch.tensor(float(self.time), dtype=time_dtype),
            "prices": torch.tensor([float(self.prices[pair]) for pair in pair_order], dtype=dtype),
            "volume": torch.tensor([float(self.volume.get(pair, 0.0)) for pair in pair_order], dtype=dtype),
        }
