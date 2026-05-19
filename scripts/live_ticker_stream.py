import asyncio
import signal
import sys

from src.kraken import KrakenLiveFeed


PAIR = sys.argv[1] if len(sys.argv) > 1 else "BTC/USD"
DEPTH = 10


async def stream(pair_like: str, depth: int = 10) -> None:
    """Stream live ticker and book updates for one pair.

    Args:
        pair_like (str): Pair symbol or code to stream.
        depth (int): Order book depth to subscribe to. Defaults to ``10``.

    Returns:
        None: This coroutine does not return a value.
    """
    async with KrakenLiveFeed([pair_like], depth=depth) as feed:
        await feed.wait_until_ready(timeout=30.0)
        print(f"Streaming {feed.pairs[0]} (depth={depth}). CTRL+C to stop.\n")

        while True:
            update = await feed.next_update()
            book = update.book

            if book and book.best_bid_price is not None and book.best_ask_price is not None:
                print(
                    f"{update.pair}  last {update.price:,.2f}  |  "
                    f"bid {book.best_bid_price:,.2f} ({book.best_bid_size:.6f})  "
                    f"ask {book.best_ask_price:,.2f} ({book.best_ask_size:.6f})  "
                    f"spread {book.spread:.2f}  mid {book.mid:,.2f}"
                )
            elif update.ticker is not None:
                print(
                    f"{update.pair}  last {update.ticker.last:,.2f}  |  "
                    f"bid {update.ticker.bid:,.2f}  ask {update.ticker.ask:,.2f}  "
                    f"spread {update.ticker.spread:.2f}  mid {update.ticker.mid:,.2f}"
                )


def _install_sigint(loop) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            pass


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_sigint(loop)
    loop.run_until_complete(stream(PAIR, DEPTH))


if __name__ == "__main__":
    main()
