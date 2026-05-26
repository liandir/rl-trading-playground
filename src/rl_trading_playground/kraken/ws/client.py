"""Client utilities for Kraken market data and exchange integration helpers."""
from __future__ import annotations

import json
from typing import Any


KRAKEN_PUBLIC_WS_URL = "wss://ws.kraken.com/"


class KrakenPublicWSClient:
    """KrakenPublicWSClient client for Kraken market data and exchange integration helpers."""
    def __init__(
        self,
        *,
        url: str = KRAKEN_PUBLIC_WS_URL,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
        max_queue: int | None = 4096,
    ) -> None:
        """Initialize the instance.

        Args:
            url (str): The url value. Defaults to ``KRAKEN_PUBLIC_WS_URL``.
            ping_interval (float): The ping interval value. Defaults to ``20.0``.
            ping_timeout (float): The ping timeout value. Defaults to ``20.0``.
            max_queue (int | None): The max queue value. Defaults to ``4096``.

        Returns:
            None: This function does not return a value.
        """
        self.url = url
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.max_queue = max_queue
        self._conn = None

    @property
    def connected(self) -> bool:
        """Connected for KrakenPublicWSClient.

        Returns:
            bool: The computed or requested result.
        """
        return self._conn is not None

    async def __aenter__(self) -> "KrakenPublicWSClient":
        """Enter the async context manager.

        Returns:
            'KrakenPublicWSClient': The computed or requested result.
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit the async context manager.

        Args:
            exc_type (Any): The exc type value.
            exc (Any): The exc value.
            tb (Any): The tb value.

        Returns:
            None: This function does not return a value.
        """
        await self.close()

    async def connect(self) -> None:
        """Open the underlying connection.

        Returns:
            None: This function does not return a value.
        """
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
        """Close resources held by the component.

        Returns:
            None: This function does not return a value.
        """
        if self._conn is None:
            return
        try:
            await self._conn.close()
        finally:
            self._conn = None

    async def send_json(self, payload: dict[str, Any]) -> None:
        """Send json for KrakenPublicWSClient.

        Args:
            payload (dict[str, Any]): The payload value.

        Returns:
            None: This function does not return a value.
        """
        if self._conn is None:
            raise RuntimeError("WebSocket is not connected.")
        await self._conn.send(json.dumps(payload))

    async def recv_json(self) -> Any:
        """Recv json for KrakenPublicWSClient.

        Returns:
            Any: The computed or requested result.
        """
        if self._conn is None:
            raise RuntimeError("WebSocket is not connected.")
        raw = await self._conn.recv()
        return json.loads(raw)

    async def subscribe(self, pairs: list[str] | tuple[str, ...], channel: str, **params: Any) -> None:
        """Subscribe to the configured data stream.

        Args:
            pairs (list[str] | tuple[str, ...]): The pairs value.
            channel (str): The channel value.
            **params (Any): The params value.

        Returns:
            None: This function does not return a value.
        """
        payload = {
            "event": "subscribe",
            "pair": list(pairs),
            "subscription": {"name": channel, **params},
        }
        await self.send_json(payload)

    async def unsubscribe(self, pairs: list[str] | tuple[str, ...], channel: str, **params: Any) -> None:
        """Unsubscribe for KrakenPublicWSClient.

        Args:
            pairs (list[str] | tuple[str, ...]): The pairs value.
            channel (str): The channel value.
            **params (Any): The params value.

        Returns:
            None: This function does not return a value.
        """
        payload = {
            "event": "unsubscribe",
            "pair": list(pairs),
            "subscription": {"name": channel, **params},
        }
        await self.send_json(payload)
