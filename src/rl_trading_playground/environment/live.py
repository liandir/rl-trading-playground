"""Live utilities for trading environment state, action, reward, and simulation logic."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from rl_trading_playground.kraken.feed import KrakenLiveFeed
from rl_trading_playground.kraken.resample import IntervalAggregator


class LiveFrameSource:
    """
    Convert a KrakenLiveFeed into the same payload shape used by the historical env.

    Each yielded payload matches:
        {"time": Tensor[()], "prices": Tensor[N], "volume": Tensor[N]}
    """

    def __init__(
        self,
        feed: KrakenLiveFeed,
        *,
        interval_seconds: float,
        pair_order: Sequence[str] | None = None,
        dtype: torch.dtype = torch.float32,
        time_dtype: torch.dtype = torch.float64,
        wait_until_all_pairs_ready: bool = True,
    ) -> None:
        """Initialize the instance.

        Args:
            feed (KrakenLiveFeed): The feed value.
            interval_seconds (float): The interval seconds value.
            pair_order (Sequence[str] | None): The pair order value. Defaults to ``None``.
            dtype (torch.dtype): The dtype value. Defaults to ``torch.float32``.
            time_dtype (torch.dtype): The time dtype value. Defaults to ``torch.float64``.
            wait_until_all_pairs_ready (bool): The wait until all pairs ready value. Defaults to ``True``.

        Returns:
            None: This function does not return a value.
        """
        self.feed = feed
        self.interval_seconds = float(interval_seconds)
        self.pair_order = tuple(pair_order or feed.pairs)
        self.dtype = dtype
        self.time_dtype = time_dtype
        self.wait_until_all_pairs_ready = bool(wait_until_all_pairs_ready)
        self._aggregator = IntervalAggregator(
            self.pair_order,
            self.interval_seconds,
            wait_until_all_pairs_ready=self.wait_until_all_pairs_ready,
        )

    async def reset(self, timeout: float | None = None) -> dict[str, torch.Tensor]:
        """Reset internal state for a new episode or stream.

        Args:
            timeout (float | None): The timeout value. Defaults to ``None``.

        Returns:
            dict[str, torch.Tensor]: The computed or requested result.
        """
        return await self.next(timeout=timeout)

    async def step(self, timeout: float | None = None) -> dict[str, torch.Tensor]:
        """Advance the simulation by one step.

        Args:
            timeout (float | None): The timeout value. Defaults to ``None``.

        Returns:
            dict[str, torch.Tensor]: The computed or requested result.
        """
        return await self.next(timeout=timeout)

    async def next(self, timeout: float | None = None) -> dict[str, torch.Tensor]:
        """Next for LiveFrameSource.

        Args:
            timeout (float | None): The timeout value. Defaults to ``None``.

        Returns:
            dict[str, torch.Tensor]: The computed or requested result.
        """
        while True:
            update = await self.feed.next_update(timeout=timeout)
            frames = self._aggregator.add(update)
            if not frames:
                continue
            frame = frames[-1]
            return frame.to_env_input(
                self.pair_order,
                dtype=self.dtype,
                time_dtype=self.time_dtype,
            )
