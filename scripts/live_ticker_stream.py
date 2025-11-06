# Live streaming ticker + order book using Kraken public WebSocket.
# Uses KrakenSpotClient only to resolve the pair symbol.

import asyncio
import json
import math
import signal
import sys
from collections import defaultdict
from typing import Dict, Tuple, List

from src.kraken import KrakenSpotClient


WS_URL = "wss://ws.kraken.com/"
PAIR = sys.argv[1] if len(sys.argv) > 1 else "BTC/USD"
DEPTH = 10   # requested book depth (Kraken supports 10/25/100/500/1000)


# ------------- tiny orderbook helper (price->size maps) -------------

class L2Book:
    def __init__(self):
        self.bids: Dict[float, float] = defaultdict(float)  # max-price first
        self.asks: Dict[float, float] = defaultdict(float)  # min-price first

    @staticmethod
    def _apply(side: Dict[float, float], updates: List[List[str]]):
        # each entry is [price, size, timestamp]
        for u in updates:
            p = float(u[0]); s = float(u[1])
            if s == 0.0:
                side.pop(p, None)
            else:
                side[p] = s

    def apply_snapshot(self, data: dict):
        if "as" in data: self._apply(self.asks, data["as"])
        if "bs" in data: self._apply(self.bids, data["bs"])

    def apply_update(self, data: dict):
        if "a" in data: self._apply(self.asks, data["a"])
        if "b" in data: self._apply(self.bids, data["b"])

    def best_bid(self) -> Tuple[float, float] | None:
        return max(self.bids.items(), key=lambda kv: kv[0]) if self.bids else None

    def best_ask(self) -> Tuple[float, float] | None:
        return min(self.asks.items(), key=lambda kv: kv[0]) if self.asks else None

    def top(self, n=5) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        bids = sorted(self.bids.items(), key=lambda kv: kv[0], reverse=True)[:n]
        asks = sorted(self.asks.items(), key=lambda kv: kv[0])[:n]
        return bids, asks


# ------------- WS logic -------------

async def stream(pair_like: str, depth: int = 10):
    # resolve to WS display name (e.g., XBT/USD)
    k = KrakenSpotClient()
    wsname = k.resolve_pair(pair_like)  # Kraken WS accepts wsname with slash

    print(f"Streaming {wsname} (depth={depth}). CTRL+C to stop.\n")

    backoff = 1.0
    while True:
        try:
            async with _connect() as ws:
                await _subscribe(ws, wsname, depth)
                book = L2Book()
                chan_ids = {}  # map channelID -> channel name
                got_snapshot = False

                while True:
                    raw = await ws.recv()
                    msg = json.loads(raw)

                    # system/event messages are dicts; data messages are lists
                    if isinstance(msg, dict):
                        et = msg.get("event")
                        if et == "subscriptionStatus":
                            status = msg.get("status")
                            chan_ids[msg["channelID"]] = msg["subscription"]["name"]
                            print(f"[status] {msg['subscription']['name']}: {status}")
                        elif et in ("heartbeat", "systemStatus"):
                            continue
                        else:
                            # unknown event, just print once
                            # print("[event]", msg)
                            pass
                        continue

                    # data message: [channelID, payload..., channelName, pair]
                    if not isinstance(msg, list) or len(msg) < 3:
                        continue

                    channel = msg[-2]
                    pair = msg[-1]

                    if channel.startswith("book"):
                        payload = msg[1]
                        if "as" in payload or "bs" in payload:
                            # snapshot
                            book.apply_snapshot(payload)
                            got_snapshot = True
                        else:
                            # updates
                            book.apply_update(payload)

                        if got_snapshot:
                            bb = book.best_bid()
                            ba = book.best_ask()
                            if bb and ba:
                                bid_px, bid_sz = bb
                                ask_px, ask_sz = ba
                                spread = ask_px - bid_px
                                mid = (ask_px + bid_px) / 2.0
                                print(f"{pair}  bid {bid_px:,.2f} ({bid_sz:.6f})  |  "
                                      f"ask {ask_px:,.2f} ({ask_sz:.6f})  "
                                      f"spread {spread:.2f}  mid {mid:,.2f}")

                    elif channel == "ticker":
                        payload = msg[1]
                        # payload fields: a (ask), b (bid), c (last), etc.
                        ask = float(payload["a"][0])
                        bid = float(payload["b"][0])
                        last = float(payload["c"][0])
                        spread = ask - bid
                        mid = (ask + bid) / 2.0
                        print(f"{pair}  last {last:,.2f}  |  bid {bid:,.2f}  "
                              f"ask {ask:,.2f}  spread {spread:.2f}  mid {mid:,.2f}")

            backoff = 1.0  # reset if clean exit
        
        except asyncio.CancelledError:
            raise

        except Exception as e:
            print(f"[reconnect] {e!r} — reconnecting in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


# context manager wrapper with ping
class _connect:
    def __init__(self):
        import websockets  # lazy import to make dependency obvious
        self._ws_mod = websockets
        self._conn = None

    async def __aenter__(self):
        self._conn = await self._ws_mod.connect(WS_URL, ping_interval=20, ping_timeout=20)
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._conn:
                await self._conn.close()
        finally:
            self._conn = None


async def _subscribe(ws, wsname: str, depth: int):
    # subscribe to ticker
    await ws.send(json.dumps({
        "event": "subscribe",
        "pair": [wsname],
        "subscription": {"name": "ticker"},
    }))
    # subscribe to order book
    await ws.send(json.dumps({
        "event": "subscribe",
        "pair": [wsname],
        "subscription": {"name": f"book", "depth": depth},
    }))


# ------------- entrypoint -------------


def _install_sigint(loop):
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            pass  # e.g., Windows


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_sigint(loop)
    loop.run_until_complete(stream(PAIR, DEPTH))


if __name__ == "__main__":
    main()
