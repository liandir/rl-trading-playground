"""Book utilities for Kraken market data and exchange integration helpers."""
from __future__ import annotations

from collections import defaultdict

from ..models import BookSnapshot


class L2Book:
    """L2Book implementation for Kraken market data and exchange integration helpers."""
    def __init__(self) -> None:
        """Initialize the instance.

        Returns:
            None: This function does not return a value.
        """
        self.bids: dict[float, float] = defaultdict(float)
        self.asks: dict[float, float] = defaultdict(float)

    @staticmethod
    def _apply(side: dict[float, float], updates: list[list[str]]) -> None:
        """Apply for L2Book.

        Args:
            side (dict[float, float]): The side value.
            updates (list[list[str]]): The updates value.

        Returns:
            None: This function does not return a value.
        """
        for row in updates:
            price = float(row[0])
            size = float(row[1])
            if size == 0.0:
                side.pop(price, None)
            else:
                side[price] = size

    def apply_snapshot(self, payload: dict) -> None:
        """Apply snapshot for L2Book.

        Args:
            payload (dict): The payload value.

        Returns:
            None: This function does not return a value.
        """
        if "as" in payload:
            self._apply(self.asks, payload["as"])
        if "bs" in payload:
            self._apply(self.bids, payload["bs"])

    def apply_update(self, payload: dict) -> None:
        """Apply update for L2Book.

        Args:
            payload (dict): The payload value.

        Returns:
            None: This function does not return a value.
        """
        if "a" in payload:
            self._apply(self.asks, payload["a"])
        if "b" in payload:
            self._apply(self.bids, payload["b"])

    def best_bid(self) -> tuple[float, float] | None:
        """Best bid for L2Book.

        Returns:
            tuple[float, float] | None: The computed or requested result.
        """
        return max(self.bids.items(), key=lambda item: item[0]) if self.bids else None

    def best_ask(self) -> tuple[float, float] | None:
        """Best ask for L2Book.

        Returns:
            tuple[float, float] | None: The computed or requested result.
        """
        return min(self.asks.items(), key=lambda item: item[0]) if self.asks else None

    def snapshot(self) -> BookSnapshot:
        """Snapshot for L2Book.

        Returns:
            BookSnapshot: The computed or requested result.
        """
        bid = self.best_bid()
        ask = self.best_ask()
        if not bid or not ask:
            return BookSnapshot()

        spread = ask[0] - bid[0]
        mid = (ask[0] + bid[0]) / 2.0
        return BookSnapshot(
            best_bid_price=bid[0],
            best_bid_size=bid[1],
            best_ask_price=ask[0],
            best_ask_size=ask[1],
            spread=spread,
            mid=mid,
        )
