"""
live_data.py — print live ticker and order book data using KrakenSpotClient.

Usage:
    python live_data.py [PAIR]
Example:
    python live_data.py BTC/USD
"""

import sys
import time
from rl_trading_playground.kraken import KrakenSpotClient

PAIR = sys.argv[1] if len(sys.argv) > 1 else "BTC/USD"
DEPTH = 5
REFRESH = 1.0  # seconds

def main():
    k = KrakenSpotClient()
    pair = k.resolve_pair(PAIR)
    print(f"Fetching live data for {PAIR} → {pair}\n(CTRL+C to stop)\n")

    while True:
        try:
            # --- get ticker info ---
            tkr = k.get_ticker(pair)
            ask = float(tkr["a"][0])
            bid = float(tkr["b"][0])
            last = float(tkr["c"][0])
            spread = ask - bid
            mid = (ask + bid) / 2.0

            # --- print concise ticker info ---
            print("=" * 70)
            print(f"{time.strftime('%H:%M:%S')}  Pair {pair}")
            print(f"Last: {last:,.5f} | Bid: {bid:,.5f} | Ask: {ask:,.5f} | "
                  f"Spread: {spread:,.5f} | Mid: {mid:,.5f}")

            # --- optional: show small snapshot of the order book ---
            ob = k.get_order_book(pair, DEPTH)
            bids = ob["bids"][:DEPTH]
            asks = ob["asks"][:DEPTH]
            print("\nTop bids:")
            for p, v, _ in bids:
                print(f"  bid {float(p):>10,.6f}  size {float(v):>10.3f}")
            print("Top asks:")
            for p, v, _ in asks:
                print(f"  ask {float(p):>10,.6f}  size {float(v):>10.3f}")

            time.sleep(REFRESH)

        except KeyboardInterrupt:
            print("\nStopped.")
            break

        except Exception as e:
            print(f"[WARN] {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
