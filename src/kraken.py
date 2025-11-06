"""
kraken_spot.py — minimal Kraken Spot client (REST)

Features
- Public:   get_ticker(pair), get_order_book(pair, depth)
- Private:  get_balance(), place_order(...), cancel_order(txid), query_orders_info(...)
- Pair resolution: resolve_pair("BTC/USD") -> "XBT/USD" (auto-maps against /public/AssetPairs)
- Idempotency: optional `userref` you can set on orders and look up later.
- Nonce: strictly increasing per-key (thread-safe).

Security
- Reads API key/secret from env: KRAKEN_API_KEY, KRAKEN_API_SECRET
- NEVER logs your secret.

Auth math (Spot private endpoints)
Let:
    P  = URI path bytes (e.g., b"/0/private/Balance")
    n  = nonce string (e.g., "1730600000000")
    q  = URL-encoded POST body (including nonce=n)
    S  = base64-decoded API secret (bytes)

Signature:
    API-Sign = base64( HMAC_SHA512(S,  P || SHA256(n || q)) )

This implements exactly that.
"""

from __future__ import annotations
import base64
import hashlib
import hmac
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from typing import Dict, Any, Optional, Tuple


KRAKEN_API_URL = "https://api.kraken.com"


class KrakenAPIError(RuntimeError):
    """Raised when Kraken returns errors or we detect a bad response."""
    pass


class _Nonce:
    """Thread-safe, strictly increasing nonce (uint64 as a string)."""
    def __init__(self):
        self._lock = threading.Lock()
        # start from current ms; guarantees monotonicity per process
        self._last = int(time.time() * 1e3)

    def next(self) -> str:
        with self._lock:
            now = int(time.time() * 1e3)
            if now <= self._last:
                self._last += 1
            else:
                self._last = now
            return str(self._last)


class KrakenSpotClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        user_agent: str = "kraken-spot-python/1.0",
        timeout: float = 10.0,
        rate_limit_per_sec: float = 15.0,  # very conservative
    ):
        self.api_key = api_key or os.getenv("KRAKEN_API_KEY", "")
        self.api_secret_b64 = api_secret or os.getenv("KRAKEN_API_SECRET", "")
        if not self.api_key or not self.api_secret_b64:
            # You may still use public endpoints without keys.
            pass

        self._timeout = timeout
        self._ua = user_agent
        self._nonce = _Nonce()
        self._asset_pairs_cache: Dict[str, Dict[str, Any]] = {}
        self._pair_alias_cache: Dict[str, str] = {}

        # naive token bucket for public endpoints (conservative)
        self._rl_lock = threading.Lock()
        self._rl_capacity = max(rate_limit_per_sec, 1.0)
        self._rl_tokens = self._rl_capacity
        self._rl_last = time.time()

    # ---------- Public helpers ----------

    def resolve_pair(self, pair_like: str) -> str:
        """
        Converts human input like "BTC/USD" or "ETH-EUR" to Kraken's canonical pair,
        e.g., "XBT/USD" or "ETH/EUR", by consulting /public/AssetPairs.
        Caches the mapping for speed.
        """
        key = pair_like.strip().upper().replace("-", "/")
        if key in self._pair_alias_cache:
            return self._pair_alias_cache[key]

        # Load catalog if needed
        if not self._asset_pairs_cache:
            cat = self._get_public("/0/public/AssetPairs")["result"]
            self._asset_pairs_cache = cat

        # direct match first
        if key in self._asset_pairs_cache:
            self._pair_alias_cache[key] = key
            return key

        # Try to normalize common symbols (BTC -> XBT)
        a, b = [s.strip() for s in key.split("/")]
        sym_map = {"BTC": "XBT", "XBT": "XBT"}
        a_norm = sym_map.get(a, a)
        candidate1 = f"{a_norm}/{b}"
        candidate2 = f"{a_norm}{b}".replace("/", "")
        # Scan catalog names and altnames
        for pname, meta in self._asset_pairs_cache.items():
            alt = meta.get("altname", "")
            wsname = meta.get("wsname", "")
            if key in (alt, wsname, pname) or candidate1 in (alt, wsname, pname) or candidate2 in (alt, wsname, pname):
                # Prefer wsname if present (e.g., "XBT/USD"), otherwise pname.
                resolved = wsname or pname
                self._pair_alias_cache[key] = resolved
                return resolved

        raise KrakenAPIError(f"Cannot resolve pair '{pair_like}'. Check /public/AssetPairs.")

    def get_ticker(self, pair_like: str) -> Dict[str, Any]:
        """
        Returns merged ticker info for the pair:
        - a: ask [price, wholeLotVolume, lotVolume]
        - b: bid [price, wholeLotVolume, lotVolume]
        - c: last trade [price, lotVolume]
        - v: volume [today, last 24h], etc.
        """
        pair = self.resolve_pair(pair_like)
        data = self._get_public("/0/public/Ticker", {"pair": pair})
        res = data["result"]
        # Kraken returns keyed by canonical alt/key; grab the first
        first_key = next(iter(res.keys()))
        out = res[first_key]
        out["_pair"] = first_key
        return out

    def get_order_book(self, pair_like: str, depth: int = 10) -> Dict[str, Any]:
        """
        Returns order book with 'asks' and 'bids', each a list of [price, volume, timestamp].
        """
        pair = self.resolve_pair(pair_like)
        depth = max(1, min(int(depth), 500))
        data = self._get_public("/0/public/Depth", {"pair": pair, "count": str(depth)})
        res = data["result"]
        first_key = next(iter(res.keys()))
        book = res[first_key]
        book["_pair"] = first_key
        return book

    # ---------- Private helpers (trading) ----------

    def get_balance(self) -> Dict[str, str]:
        return self._post_private("/0/private/Balance", {} )["result"]

    def place_order(
        self,
        pair_like: str,
        side: str,                 # "buy" or "sell"
        ordertype: str = "market", # "market" | "limit" | "stop-loss" | "take-profit" | ...
        volume: float = 0.0,       # base-asset amount
        price: Optional[float] = None,  # needed for limit/stop/TP variants
        *,
        userref: Optional[int] = None,  # idempotency key you choose (int32)
        timeinforce: Optional[str] = None,  # "GTC" | "IOC" | "GTD"
        starttm: Optional[str] = None,      # scheduled start time (unix sec or '+'offset)
        expiretm: Optional[str] = None,     # scheduled expiry
        validate: bool = False,             # True = dry-run
        oflags: Optional[str] = None,       # e.g., "post" (post-only), "fcib", "fciq", "reduce-only"
        leverage: Optional[str] = None,     # for margin trading (e.g., "2:1")
        trading_agreement: bool = True,     # Kraken requires explicit "agree" on first trade
    ) -> Dict[str, Any]:
        """
        Places an order and returns Kraken's response:
        - txid(s) in result["txid"]
        - description in result["descr"]["order"]
        """
        pair = self.resolve_pair(pair_like)
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        ordertype = ordertype.lower()

        payload = {
            "pair": pair.replace("/", ""),
            "type": side,
            "ordertype": ordertype,
            "volume": str(volume),
        }
        if price is not None:
            payload["price"] = _fmt_num(price)
        if userref is not None:
            payload["userref"] = str(int(userref))
        if timeinforce:
            payload["timeinforce"] = timeinforce
        if starttm:
            payload["starttm"] = str(starttm)
        if expiretm:
            payload["expiretm"] = str(expiretm)
        if validate:
            payload["validate"] = "true"
        if oflags:
            payload["oflags"] = oflags
        if leverage:
            payload["leverage"] = leverage
        if trading_agreement:
            payload["trading_agreement"] = "agree"

        return self._post_private("/0/private/AddOrder", payload)["result"]

    def cancel_order(self, txid: str) -> Dict[str, Any]:
        return self._post_private("/0/private/CancelOrder", {"txid": txid})["result"]

    def query_orders_info(
        self,
        txid: Optional[str] = None,
        *,
        userref: Optional[int] = None,
        trades: bool = False,
        consolidate_taker: bool = False,
    ) -> Dict[str, Any]:
        """
        Query by txid (comma-separated) or by userref.
        """
        params: Dict[str, str] = {}
        if txid:
            params["txid"] = txid
        if userref is not None:
            params["userref"] = str(int(userref))
        if trades:
            params["trades"] = "true"
        if consolidate_taker:
            params["consolidate_taker"] = "true"
        return self._post_private("/0/private/QueryOrders", params)["result"]

    # ---------- Low-level HTTP ----------

    def _rate_limit(self):
        # very simple token bucket for public GETs
        with self._rl_lock:
            now = time.time()
            elapsed = now - self._rl_last
            self._rl_last = now
            self._rl_tokens = min(self._rl_capacity, self._rl_tokens + elapsed * self._rl_capacity)
            if self._rl_tokens < 1.0:
                # sleep enough to get 1 token
                need = (1.0 - self._rl_tokens) / self._rl_capacity
                time.sleep(need)
                self._rl_tokens = 0.0
            else:
                self._rl_tokens -= 1.0

    def _get_public(self, path: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        self._rate_limit()
        url = KRAKEN_API_URL + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": self._ua})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        _raise_if_error(data)
        return data

    def _post_private(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key or not self.api_secret_b64:
            raise KrakenAPIError("Missing API credentials. Set KRAKEN_API_KEY and KRAKEN_API_SECRET.")
        nonce = self._nonce.next()
        data = {"nonce": nonce, **{k: str(v) for k, v in params.items()}}
        body = urllib.parse.urlencode(data)
        url = KRAKEN_API_URL + path

        # ---- signature: base64( HMAC_SHA512( base64_dec(secret),  path || SHA256(nonce+body) ) )
        sha256 = hashlib.sha256((nonce + body).encode("utf-8")).digest()
        message = path.encode("utf-8") + sha256
        secret = base64.b64decode(self.api_secret_b64)
        sig = hmac.new(secret, message, hashlib.sha512).digest()
        api_sign = base64.b64encode(sig).decode()

        headers = {
            "User-Agent": self._ua,
            "API-Key": self.api_key,
            "API-Sign": api_sign,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        req = urllib.request.Request(url, data=body.encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        _raise_if_error(data)
        return data

# ---------- utils ----------

def _fmt_num(x: float) -> str:
    # Kraken accepts stringified decimals; keep a reasonable precision
    return f"{x:.12f}".rstrip("0").rstrip(".")

def _raise_if_error(payload: Dict[str, Any]):
    # Kraken uses {"error": [...], "result": {...}}
    if not isinstance(payload, dict) or "error" not in payload:
        raise KrakenAPIError(f"Malformed response: {payload!r}")
    errs = payload.get("error") or []
    if errs:
        # The list may include warnings (prefixed "W:"); raise if any errors ("E:")
        fatal = [e for e in errs if isinstance(e, str) and e.startswith("E")]
        if fatal:
            raise KrakenAPIError(f"Kraken error(s): {fatal} | full={errs}")

# ---------- example usage ----------

if __name__ == "__main__":
    # Demo (PUBLIC): live ticker + order book
    k = KrakenSpotClient()
    pair = "BTC/USD"
    print("Resolved pair:", k.resolve_pair(pair))
    print("Ticker:", k.get_ticker(pair))
    ob = k.get_order_book(pair, depth=5)
    print("Best Bid/Ask:", ob["bids"][0], ob["asks"][0])

    # Demo (PRIVATE): uncomment to trade (DANGEROUS — this sends real orders!)
    # Ensure env vars are set and consider validate=True first.
    # k = KrakenSpotClient()  # with keys in env
    # print("Balance:", k.get_balance())
    # # DRY-RUN a limit buy 0.001 BTC at $10,000 (won’t execute today, but shows validation)
    # res = k.place_order("BTC/USD", "buy", ordertype="limit", volume=0.001, price=10000, validate=True, userref=123456)
    # print("Validate AddOrder:", res)
    # # REAL market sell example (COMMENTED OUT!)
    # # res2 = k.place_order("BTC/USD", "sell", ordertype="market", volume=0.001, userref=123457)
    # # print("Market sell:", res2)
