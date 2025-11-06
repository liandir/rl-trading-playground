# multi_ws.py
# Stream multiple pairs over one Kraken public WebSocket and deliver a single merged snapshot object.

import asyncio
import json
import copy
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from src.kraken import KrakenSpotClient

WS_URL = "wss://ws.kraken.com/"

# --------- tiny L2 top-of-book helper ---------

class L2Book:
    def __init__(self):
        # price -> size
        self.bids: Dict[float, float] = defaultdict(float)
        self.asks: Dict[float, float] = defaultdict(float)

    @staticmethod
    def _apply(side: Dict[float, float], updates: List[List[str]]):
        # updates are lists like ["price","size","ts"] or ["price","size","ts","flag"]
        for u in updates:
            p = float(u[0]); s = float(u[1])
            if s == 0.0:
                side.pop(p, None)
            else:
                side[p] = s

    def apply_snapshot(self, payload: dict):
        if "as" in payload: self._apply(self.asks, payload["as"])
        if "bs" in payload: self._apply(self.bids, payload["bs"])

    def apply_update(self, payload: dict):
        if "a" in payload: self._apply(self.asks, payload["a"])
        if "b" in payload: self._apply(self.bids, payload["b"])

    def best_bid(self) -> Optional[Tuple[float, float]]:
        return max(self.bids.items(), key=lambda kv: kv[0]) if self.bids else None

    def best_ask(self) -> Optional[Tuple[float, float]]:
        return min(self.asks.items(), key=lambda kv: kv[0]) if self.asks else None

# --------- multi-pair stream manager ---------

class KrakenWSMulti:
    """
    Usage (async):
        async with KrakenWSMulti(["BTC/USD","ETH/EUR"], depth=10) as m:
            snap = await m.next_snapshot(timeout=2.0)  # dict of all pairs

    Snapshot format:
        {
          "XBT/USD": {"ticker": {"last":..., "bid":..., "ask":...},
                      "book": {"best_bid": [px, sz], "best_ask": [px, sz], "spread":..., "mid":...}},
          "ETH/EUR": {...}
        }
    """
    def __init__(self, pairs_like: List[str], depth: int = 10):
        self._client = KrakenSpotClient()
        self._pairs_ws = [self._client.resolve_pair(p) for p in pairs_like]  # wsname e.g. XBT/USD
        self._depth = depth
        self._conn = None
        self._task = None
        self._stop = asyncio.Event()

        # shared state
        self._books: Dict[str, L2Book] = {wsname: L2Book() for wsname in self._pairs_ws}
        self._ticker: Dict[str, Dict[str, float]] = {wsname: {"last": None, "bid": None, "ask": None}
                                                     for wsname in self._pairs_ws}

        # whenever any pair updates, we set this event to let waiters grab a merged snapshot
        self._updated = asyncio.Event()

    async def __aenter__(self):
        import websockets
        self._conn = await websockets.connect(WS_URL, ping_interval=20, ping_timeout=20)
        await self._subscribe_all()
        self._task = asyncio.create_task(self._recv_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        if self._conn:
            await self._conn.close()

    async def _subscribe_all(self):
        # subscribe ticker for all pairs at once
        await self._conn.send(json.dumps({
            "event": "subscribe",
            "pair": self._pairs_ws,
            "subscription": {"name": "ticker"},
        }))
        # subscribe order book for all pairs (same connection)
        await self._conn.send(json.dumps({
            "event": "subscribe",
            "pair": self._pairs_ws,
            "subscription": {"name": "book", "depth": self._depth},
        }))

    async def _recv_loop(self):
        while not self._stop.is_set():
            try:
                raw = await self._conn.recv()
                msg = json.loads(raw)

                # system events
                if isinstance(msg, dict):
                    if msg.get("event") == "subscriptionStatus":
                        # could print/log status here if you like
                        pass
                    continue

                # data: [channelID, payload, channelName, pair]
                if not (isinstance(msg, list) and len(msg) >= 3):
                    continue
                channel = msg[-2]
                pair = msg[-1]  # wsname like "XBT/USD"
                payload = msg[1]

                if channel.startswith("book"):
                    # snapshot vs update
                    if "as" in payload or "bs" in payload:
                        self._books[pair].apply_snapshot(payload)
                    else:
                        self._books[pair].apply_update(payload)
                    self._updated.set()

                elif channel == "ticker":
                    # payload: {"a":[ask,...], "b":[bid,...], "c":[last,...], ...}
                    t = self._ticker[pair]
                    t["ask"] = float(payload["a"][0])
                    t["bid"] = float(payload["b"][0])
                    t["last"] = float(payload["c"][0])
                    self._updated.set()

            except asyncio.CancelledError:
                break
            except Exception:
                # you may want to add reconnect logic here if needed
                continue

    def _build_snapshot(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for pair in self._pairs_ws:
            book = self._books[pair]
            bb = book.best_bid()
            ba = book.best_ask()
            entry = {"ticker": copy.deepcopy(self._ticker[pair]), "book": {}}
            if bb and ba:
                spread = ba[0] - bb[0]
                mid = (ba[0] + bb[0]) / 2.0
                entry["book"] = {
                    "best_bid": [bb[0], bb[1]],
                    "best_ask": [ba[0], ba[1]],
                    "spread": spread,
                    "mid": mid,
                }
            out[pair] = entry
        return out

    async def next_snapshot(self, timeout: float = 1.0) -> Dict[str, dict]:
        """
        Await until *any* pair updates (or timeout), then return a deep-copied merged snapshot.
        """
        try:
            await asyncio.wait_for(self._updated.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # no new updates; still return current snapshot
            pass
        self._updated.clear()
        return self._build_snapshot()

# --------- convenience: synchronous, one-shot snapshot ---------

def get_snapshot_sync(pairs: List[str], timeout: float = 1.0) -> Dict[str, dict]:
    """
    Blocking helper: connect, wait for first updates (up to timeout), return merged snapshot, then close.
    Good for "give me the latest for N pairs *now*".
    """
    async def _runner():
        async with KrakenWSMulti(pairs) as m:
            return await m.next_snapshot(timeout=timeout)
    return asyncio.run(_runner())

# --------- demo ---------

if __name__ == "__main__":
    # Example: stream continuously and print snapshots
    async def demo():
        pairs = ["BTC/USD", "ETH/EUR", "SOL/USD"]
        async with KrakenWSMulti(pairs, depth=10) as m:
            while True:
                snap = await m.next_snapshot(timeout=2.0)
                print(snap)  # a single object with all pairs' ticker+book tops

    asyncio.run(demo())
