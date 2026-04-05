from __future__ import annotations

import math

from .models import MarketFrame, PairSnapshot


class IntervalAggregator:
    def __init__(
        self,
        pairs: list[str] | tuple[str, ...],
        interval_seconds: float,
        *,
        wait_until_all_pairs_ready: bool = True,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive.")

        self.pairs = tuple(pairs)
        self.interval_seconds = float(interval_seconds)
        self.wait_until_all_pairs_ready = wait_until_all_pairs_ready

        self._bucket_start: float | None = None
        self._last_prices: dict[str, float] = {}
        self._bucket_volume: dict[str, float] = {pair: 0.0 for pair in self.pairs}
        self._ready_pairs: set[str] = set()

    def add(self, update: PairSnapshot) -> list[MarketFrame]:
        event_time = float(update.timestamp)
        if self._bucket_start is None:
            self._bucket_start = math.floor(event_time / self.interval_seconds) * self.interval_seconds

        emitted: list[MarketFrame] = []
        while event_time >= self._bucket_start + self.interval_seconds:
            frame = self._build_frame(self._bucket_start + self.interval_seconds)
            if frame is not None:
                emitted.append(frame)
            self._bucket_start += self.interval_seconds
            self._bucket_volume = {pair: 0.0 for pair in self.pairs}

        if update.price is not None:
            self._last_prices[update.pair] = float(update.price)
            self._ready_pairs.add(update.pair)

        if update.trade_volume > 0.0:
            self._bucket_volume[update.pair] = self._bucket_volume.get(update.pair, 0.0) + float(update.trade_volume)

        return emitted

    def flush(self) -> MarketFrame | None:
        if self._bucket_start is None:
            return None
        return self._build_frame(self._bucket_start + self.interval_seconds)

    def _build_frame(self, frame_time: float) -> MarketFrame | None:
        if self.wait_until_all_pairs_ready and len(self._ready_pairs) < len(self.pairs):
            return None

        missing = [pair for pair in self.pairs if pair not in self._last_prices]
        if missing:
            return None

        return MarketFrame(
            time=float(frame_time),
            prices={pair: float(self._last_prices[pair]) for pair in self.pairs},
            volume={pair: float(self._bucket_volume.get(pair, 0.0)) for pair in self.pairs},
        )
