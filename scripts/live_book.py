# live_book.py (fixed)
# Polls Kraken order book (public REST) and prints top-of-book, spread, and mid.
# Uses the REST "pair code" (e.g., XBTUSD), not wsname (XBT/USD).

import json
import time
import urllib.parse
import urllib.request

API = "https://api.kraken.com"

def load_catalog():
    url = API + "/0/public/AssetPairs"
    req = urllib.request.Request(url, headers={"User-Agent": "kraken-live-book/1.1"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("error"):
        raise RuntimeError(f"Kraken error: {payload['error']}")
    return payload["result"]  # dict: {pair_code: meta}

def resolve_pair_codes(pair_like: str):
    """
    Return a tuple: (api_pair_code_for_rest, human_wsname_for_printing).
    - Input like 'BTC/USD', 'XBTUSD', 'eth-eur' all okay.
    - We prefer REST altname (e.g., 'XBTUSD'); if missing, fall back to the dict key.

    Args:
        pair_like (str): Pair symbol or code to resolve.

    Returns:
        tuple: REST pair code and human-readable display name.
    """
    pair_like = pair_like.upper().replace("-", "/").strip()
    base, quote = [p.strip() for p in pair_like.split("/")] if "/" in pair_like else (pair_like, None)
    # Normalize BTC -> XBT for Kraken
    if base == "BTC":
        base = "XBT"
    catalog = load_catalog()

    # Try exact by altname or key if user passed a code without slash
    if quote is None:
        for k, m in catalog.items():
            if pair_like in (m.get("altname", ""), k):
                return (m.get("altname") or k, m.get("wsname") or m.get("altname") or k)

    # Build candidates
    if quote is None:
        raise RuntimeError(f"Cannot parse pair '{pair_like}'. Try like 'BTC/USD' or 'XBTUSD'.")
    candidates_no_slash = (base + quote, )
    candidates_with_slash = (f"{base}/{quote}", )

    # Pass 1: exact matches on altname or key (REST-acceptable)
    for k, m in catalog.items():
        alt = m.get("altname", "")
        if alt in candidates_no_slash or k in candidates_no_slash:
            return (alt or k, m.get("wsname") or alt or k)

    # Pass 2: matches on wsname (for user friendliness), then return REST code
    for k, m in catalog.items():
        ws = m.get("wsname", "")
        alt = m.get("altname", "")
        if ws in candidates_with_slash:
            return (alt or k, ws)
        if ws == f"{base} / {quote}":  # some wsname include spaces around slash
            return (alt or k, ws)

    # Pass 3: looser scan
    for k, m in catalog.items():
        alt = m.get("altname", "")
        ws = m.get("wsname", "")
        if pair_like in (ws, alt, k):
            return (alt or k, ws or alt or k)

    raise RuntimeError(f"Cannot resolve pair '{pair_like}'. Is it listed on Kraken Spot?")

def fetch_order_book(api_pair_code: str, depth: int = 5):
    """Return order book rows for a REST pair code.

    Args:
        api_pair_code (str): Kraken REST pair code like ``XBTUSD``.
        depth (int): Number of rows to request. Defaults to ``5``.

    Returns:
        tuple: Response key, bid rows, and ask rows.
    """
    params = {"pair": api_pair_code, "count": str(int(depth))}
    url = API + "/0/public/Depth?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "kraken-live-book/1.1"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("error"):
        raise RuntimeError(f"Kraken error: {payload['error']}")
    data = payload["result"]
    key = next(iter(data.keys()))
    book = data[key]
    bids = [[float(p), float(v), int(t)] for (p, v, t) in book["bids"]]
    asks = [[float(p), float(v), int(t)] for (p, v, t) in book["asks"]]
    return key, bids, asks

def fmt_row(side, row):
    """Format one order book row for terminal output.

    Args:
        side (str): Display label for the row side.
        row (list): Parsed order book row.

    Returns:
        str: Formatted row text.
    """
    return f"{side:<4} {row[0]:>12,.2f}  size {row[1]:>10.6f}"

def main(pair_like: str = "BTC/USD", depth: int = 5, interval_sec: float = 1.0):
    """Poll and print a live order book summary.

    Args:
        pair_like (str): Pair symbol or code to display. Defaults to ``"BTC/USD"``.
        depth (int): Number of book rows to print. Defaults to ``5``.
        interval_sec (float): Delay between polls in seconds. Defaults to ``1.0``.

    Returns:
        None: This function does not return a value.
    """
    api_code, pretty = resolve_pair_codes(pair_like)
    print(f"Resolved '{pair_like}' → REST code '{api_code}'  (display: {pretty})\n(CTRL+C to stop)\n")
    
    while True:
        key, bids, asks = fetch_order_book(api_code, depth)
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid
        mid = (best_ask + best_bid) / 2.0

        print("=" * 60)
        print(f"Pair: {pretty} [{key}] | Best Bid: {best_bid:,.2f} | Best Ask: {best_ask:,.2f} | "
              f"Spread: {spread:,.2f} | Mid: {mid:,.2f}")
        print("- BIDS (top)")
        for r in bids[:depth]:
            print(fmt_row("bid", r))
        print("- ASKS (top)")
        for r in asks[:depth]:
            print(fmt_row("ask", r))
        time.sleep(interval_sec)

if __name__ == "__main__":
    # Try other pairs too: "ETH/EUR", "SOL/USD", "USDT/USD"
    main("BTC/EUR", depth=5, interval_sec=1.0)
