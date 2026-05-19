import asyncio

from src.kraken import KrakenLiveFeed


async def demo() -> None:
    pairs = ["BTC/USD", "ETH/EUR", "SOL/USD"]
    async with KrakenLiveFeed(pairs, depth=10) as feed:
        await feed.wait_until_ready(timeout=30.0)
        while True:
            snapshot = await feed.next_snapshot(timeout=2.0)
            compact = {}
            for pair, state in snapshot.items():
                compact[pair] = {
                    "last": state.price,
                    "bid": state.book.best_bid_price if state.book else None,
                    "ask": state.book.best_ask_price if state.book else None,
                    "mid": state.book.mid if state.book else None,
                }
            print(compact)


def get_snapshot_sync(pairs, timeout: float = 5.0):
    """Return one live snapshot for multiple pairs.

    Args:
        pairs (list): Pair symbols or codes to stream.
        timeout (float): Maximum wait time in seconds. Defaults to ``5.0``.

    Returns:
        dict: Latest pair snapshots keyed by pair.
    """
    async def _runner():
        async with KrakenLiveFeed(pairs, depth=10) as feed:
            await feed.wait_until_ready(timeout=timeout)
            return await feed.next_snapshot(timeout=timeout)

    return asyncio.run(_runner())


if __name__ == "__main__":
    asyncio.run(demo())
