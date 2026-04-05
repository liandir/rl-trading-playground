from __future__ import annotations

import asyncio
import copy
import time

from .models import PairSnapshot, TickerSnapshot, TradeEvent
from .resample import IntervalAggregator
from .rest import KrakenSpotClient
from .ws.book import L2Book
from .ws.client import KrakenPublicWSClient


class KrakenLiveFeed:
    def __init__(
        self,
        pairs_like: list[str] | tuple[str, ...],
        *,
        depth: int = 10,
        include_ticker: bool = True,
        include_book: bool = True,
        include_trades: bool = True,
        queue_size: int = 4096,
        reconnect_initial: float = 1.0,
        reconnect_max: float = 30.0,
        rest_client: KrakenSpotClient | None = None,
    ) -> None:
        if not pairs_like:
            raise ValueError("pairs_like must not be empty.")

        self._rest = rest_client or KrakenSpotClient()
        self.pairs = tuple(self._rest.resolve_ws_pair(pair) for pair in pairs_like)
        self.depth = int(depth)
        self.include_ticker = bool(include_ticker)
        self.include_book = bool(include_book)
        self.include_trades = bool(include_trades)
        self.queue_size = int(queue_size)
        self.reconnect_initial = float(reconnect_initial)
        self.reconnect_max = float(reconnect_max)

        self._updates: asyncio.Queue[PairSnapshot] = asyncio.Queue(maxsize=self.queue_size)
        self._snapshot_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._last_error: Exception | None = None

        self._books = {pair: L2Book() for pair in self.pairs}
        self._state = {
            pair: PairSnapshot(pair=pair, timestamp=0.0)
            for pair in self.pairs
        }

    async def __aenter__(self) -> "KrakenLiveFeed":
        self._stop.clear()
        self._task = asyncio.create_task(self._run_forever())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def current_snapshot(self) -> dict[str, PairSnapshot]:
        return {pair: copy.deepcopy(snapshot) for pair, snapshot in self._state.items()}

    async def wait_until_ready(self, timeout: float | None = None) -> None:
        await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)

    async def next_update(self, timeout: float | None = None) -> PairSnapshot:
        if timeout is None:
            return await self._updates.get()
        return await asyncio.wait_for(self._updates.get(), timeout=timeout)

    async def next_snapshot(self, timeout: float | None = None) -> dict[str, PairSnapshot]:
        if timeout is None:
            await self._snapshot_event.wait()
        else:
            await asyncio.wait_for(self._snapshot_event.wait(), timeout=timeout)
        self._snapshot_event.clear()
        return self.current_snapshot()

    async def frames(
        self,
        interval_seconds: float,
        *,
        timeout: float | None = None,
        wait_until_all_pairs_ready: bool = True,
    ):
        aggregator = IntervalAggregator(
            self.pairs,
            interval_seconds,
            wait_until_all_pairs_ready=wait_until_all_pairs_ready,
        )
        while True:
            update = await self.next_update(timeout=timeout)
            for frame in aggregator.add(update):
                yield frame

    async def _run_forever(self) -> None:
        backoff = self.reconnect_initial

        while not self._stop.is_set():
            self._books = {pair: L2Book() for pair in self.pairs}
            try:
                async with KrakenPublicWSClient() as ws:
                    await self._subscribe(ws)
                    backoff = self.reconnect_initial

                    while not self._stop.is_set():
                        message = await ws.recv_json()
                        await self._handle_message(message)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = exc
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, self.reconnect_max)

    async def _subscribe(self, ws: KrakenPublicWSClient) -> None:
        if self.include_ticker:
            await ws.subscribe(self.pairs, "ticker")
        if self.include_book:
            await ws.subscribe(self.pairs, "book", depth=self.depth)
        if self.include_trades:
            await ws.subscribe(self.pairs, "trade")

    async def _handle_message(self, message) -> None:
        if isinstance(message, dict):
            event = message.get("event")
            if event in {"systemStatus", "heartbeat", "subscriptionStatus"}:
                return
            return

        if not isinstance(message, list) or len(message) < 4:
            return

        channel = message[-2]
        pair = message[-1]
        payload = message[1]

        if pair not in self._state:
            return

        if channel == "ticker":
            self._handle_ticker(pair, payload)
        elif channel.startswith("book"):
            self._handle_book(pair, payload)
        elif channel == "trade":
            self._handle_trade(pair, payload)

    def _handle_ticker(self, pair: str, payload: dict) -> None:
        timestamp = time.time()
        ticker = TickerSnapshot(
            bid=float(payload["b"][0]),
            ask=float(payload["a"][0]),
            last=float(payload["c"][0]),
            volume_today=_safe_float(payload.get("v", [None, None])[0]),
            volume_24h=_safe_float(payload.get("v", [None, None])[1]),
            timestamp=timestamp,
        )
        state = self._state[pair]
        state.ticker = ticker
        state.timestamp = timestamp
        self._emit_update(pair)

    def _handle_book(self, pair: str, payload: dict) -> None:
        book = self._books[pair]
        if "as" in payload or "bs" in payload:
            book.apply_snapshot(payload)
        else:
            book.apply_update(payload)

        timestamp = _book_timestamp(payload, default=time.time())
        state = self._state[pair]
        state.book = book.snapshot()
        state.timestamp = timestamp
        self._emit_update(pair)

    def _handle_trade(self, pair: str, payload: list[list[str]]) -> None:
        trades: list[TradeEvent] = []
        total_volume = 0.0

        for row in payload:
            price = float(row[0])
            volume = float(row[1])
            timestamp = float(row[2])
            side = row[3]
            order_type = row[4]
            trades.append(
                TradeEvent(
                    pair=pair,
                    price=price,
                    volume=volume,
                    timestamp=timestamp,
                    side=side,
                    order_type=order_type,
                )
            )
            total_volume += volume

        if not trades:
            return

        state = self._state[pair]
        state.last_trade = trades[-1]
        state.timestamp = trades[-1].timestamp
        self._emit_update(pair, trade_volume=total_volume, trade_count=len(trades))

    def _emit_update(self, pair: str, *, trade_volume: float = 0.0, trade_count: int = 0) -> None:
        state = self._state[pair]
        update = state.copy()
        update.trade_volume = float(trade_volume)
        update.trade_count = int(trade_count)

        if update.price is not None and all(snapshot.price is not None for snapshot in self._state.values()):
            self._ready_event.set()

        while self._updates.full():
            try:
                self._updates.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._updates.put_nowait(update)
        self._snapshot_event.set()


def _safe_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _book_timestamp(payload: dict, *, default: float) -> float:
    timestamps: list[float] = []
    for key in ("a", "b", "as", "bs"):
        for row in payload.get(key, []):
            if len(row) >= 3:
                try:
                    timestamps.append(float(row[2]))
                except (TypeError, ValueError):
                    continue
    return max(timestamps) if timestamps else float(default)
