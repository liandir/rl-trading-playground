from __future__ import annotations

import json
from typing import Any


KRAKEN_PUBLIC_WS_URL = "wss://ws.kraken.com/"


class KrakenPublicWSClient:
    def __init__(
        self,
        *,
        url: str = KRAKEN_PUBLIC_WS_URL,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
        max_queue: int | None = 4096,
    ) -> None:
        self.url = url
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.max_queue = max_queue
        self._conn = None

    @property
    def connected(self) -> bool:
        return self._conn is not None

    async def __aenter__(self) -> "KrakenPublicWSClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._conn is not None:
            return

        import websockets

        self._conn = await websockets.connect(
            self.url,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            max_queue=self.max_queue,
        )

    async def close(self) -> None:
        if self._conn is None:
            return
        try:
            await self._conn.close()
        finally:
            self._conn = None

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self._conn is None:
            raise RuntimeError("WebSocket is not connected.")
        await self._conn.send(json.dumps(payload))

    async def recv_json(self) -> Any:
        if self._conn is None:
            raise RuntimeError("WebSocket is not connected.")
        raw = await self._conn.recv()
        return json.loads(raw)

    async def subscribe(self, pairs: list[str] | tuple[str, ...], channel: str, **params: Any) -> None:
        payload = {
            "event": "subscribe",
            "pair": list(pairs),
            "subscription": {"name": channel, **params},
        }
        await self.send_json(payload)

    async def unsubscribe(self, pairs: list[str] | tuple[str, ...], channel: str, **params: Any) -> None:
        payload = {
            "event": "unsubscribe",
            "pair": list(pairs),
            "subscription": {"name": channel, **params},
        }
        await self.send_json(payload)
