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
from typing import Any


KRAKEN_API_URL = "https://api.kraken.com"


class KrakenAPIError(RuntimeError):
    pass


class _Nonce:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = int(time.time() * 1e3)

    def next(self) -> str:
        with self._lock:
            now = int(time.time() * 1e3)
            self._last = max(self._last + 1, now)
            return str(self._last)


class KrakenSpotClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        *,
        user_agent: str = "kraken-spot-python/2.0",
        timeout: float = 10.0,
        rate_limit_per_sec: float = 15.0,
    ) -> None:
        self.api_key = api_key or os.getenv("KRAKEN_API_KEY", "")
        self.api_secret_b64 = api_secret or os.getenv("KRAKEN_API_SECRET", "")
        self._timeout = float(timeout)
        self._ua = user_agent
        self._nonce = _Nonce()
        self._asset_pairs_cache: dict[str, dict[str, Any]] = {}
        self._pair_alias_cache: dict[tuple[str, str], str] = {}

        self._rl_lock = threading.Lock()
        self._rl_capacity = max(float(rate_limit_per_sec), 1.0)
        self._rl_tokens = self._rl_capacity
        self._rl_last = time.time()

    def resolve_pair(self, pair_like: str) -> str:
        return self.resolve_ws_pair(pair_like)

    def resolve_ws_pair(self, pair_like: str) -> str:
        pair_code, meta = self._find_pair_meta(pair_like)
        return meta.get("wsname") or meta.get("altname") or pair_code

    def resolve_rest_pair(self, pair_like: str) -> str:
        pair_code, meta = self._find_pair_meta(pair_like)
        return meta.get("altname") or pair_code

    def get_pair_metadata(self, pair_like: str) -> dict[str, Any]:
        _, meta = self._find_pair_meta(pair_like)
        return dict(meta)

    def get_ticker(self, pair_like: str) -> dict[str, Any]:
        pair = self.resolve_rest_pair(pair_like)
        data = self._get_public("/0/public/Ticker", {"pair": pair})
        result = data["result"]
        first_key = next(iter(result.keys()))
        ticker = result[first_key]
        ticker["_pair"] = first_key
        return ticker

    def get_order_book(self, pair_like: str, depth: int = 10) -> dict[str, Any]:
        pair = self.resolve_rest_pair(pair_like)
        depth = max(1, min(int(depth), 500))
        data = self._get_public("/0/public/Depth", {"pair": pair, "count": str(depth)})
        result = data["result"]
        first_key = next(iter(result.keys()))
        book = result[first_key]
        book["_pair"] = first_key
        return book

    def get_balance(self) -> dict[str, str]:
        return self._post_private("/0/private/Balance", {})["result"]

    def place_order(
        self,
        pair_like: str,
        side: str,
        ordertype: str = "market",
        volume: float = 0.0,
        price: float | None = None,
        *,
        userref: int | None = None,
        timeinforce: str | None = None,
        starttm: str | None = None,
        expiretm: str | None = None,
        validate: bool = False,
        oflags: str | None = None,
        leverage: str | None = None,
        trading_agreement: bool = True,
    ) -> dict[str, Any]:
        pair = self.resolve_rest_pair(pair_like)
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        payload: dict[str, str] = {
            "pair": pair,
            "type": side,
            "ordertype": ordertype.lower(),
            "volume": _fmt_num(volume),
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

    def cancel_order(self, txid: str) -> dict[str, Any]:
        return self._post_private("/0/private/CancelOrder", {"txid": txid})["result"]

    def query_orders_info(
        self,
        txid: str | None = None,
        *,
        userref: int | None = None,
        trades: bool = False,
        consolidate_taker: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, str] = {}
        if txid:
            params["txid"] = txid
        if userref is not None:
            params["userref"] = str(int(userref))
        if trades:
            params["trades"] = "true"
        if consolidate_taker:
            params["consolidate_taker"] = "true"
        return self._post_private("/0/private/QueryOrders", params)["result"]

    def _find_pair_meta(self, pair_like: str) -> tuple[str, dict[str, Any]]:
        if not self._asset_pairs_cache:
            self._asset_pairs_cache = self._get_public("/0/public/AssetPairs")["result"]

        normalized = _normalize_pair_like(pair_like)
        candidates = _pair_candidates(normalized)

        for prefer in ("wsname", "rest"):
            cache_key = (prefer, normalized)
            cached = self._pair_alias_cache.get(cache_key)
            if cached and cached in self._asset_pairs_cache:
                return cached, self._asset_pairs_cache[cached]

        for pair_code, meta in self._asset_pairs_cache.items():
            names = {
                _normalize_pair_like(pair_code),
                _normalize_pair_like(meta.get("altname", "")),
                _normalize_pair_like(meta.get("wsname", "")),
            }
            names_no_slash = {name.replace("/", "") for name in names if name}
            if candidates & names:
                self._pair_alias_cache[("wsname", normalized)] = pair_code
                self._pair_alias_cache[("rest", normalized)] = pair_code
                return pair_code, meta
            if {candidate.replace("/", "") for candidate in candidates} & names_no_slash:
                self._pair_alias_cache[("wsname", normalized)] = pair_code
                self._pair_alias_cache[("rest", normalized)] = pair_code
                return pair_code, meta

        raise KrakenAPIError(f"Cannot resolve pair '{pair_like}'. Check /0/public/AssetPairs.")

    def _rate_limit(self) -> None:
        with self._rl_lock:
            now = time.time()
            elapsed = now - self._rl_last
            self._rl_last = now
            self._rl_tokens = min(self._rl_capacity, self._rl_tokens + elapsed * self._rl_capacity)
            if self._rl_tokens < 1.0:
                delay = (1.0 - self._rl_tokens) / self._rl_capacity
                time.sleep(delay)
                self._rl_tokens = 0.0
            else:
                self._rl_tokens -= 1.0

    def _get_public(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        self._rate_limit()
        url = KRAKEN_API_URL + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": self._ua})
        with urllib.request.urlopen(req, timeout=self._timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        _raise_if_error(payload)
        return payload

    def _post_private(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key or not self.api_secret_b64:
            raise KrakenAPIError("Missing API credentials. Set KRAKEN_API_KEY and KRAKEN_API_SECRET.")

        nonce = self._nonce.next()
        data = {"nonce": nonce, **{key: str(value) for key, value in params.items()}}
        body = urllib.parse.urlencode(data)
        url = KRAKEN_API_URL + path

        sha256 = hashlib.sha256((nonce + body).encode("utf-8")).digest()
        message = path.encode("utf-8") + sha256
        secret = base64.b64decode(self.api_secret_b64)
        signature = hmac.new(secret, message, hashlib.sha512).digest()
        api_sign = base64.b64encode(signature).decode("utf-8")

        headers = {
            "User-Agent": self._ua,
            "API-Key": self.api_key,
            "API-Sign": api_sign,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        req = urllib.request.Request(url, data=body.encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        _raise_if_error(payload)
        return payload


def _normalize_pair_like(value: str) -> str:
    return value.strip().upper().replace("-", "/").replace(" ", "")


def _pair_candidates(normalized: str) -> set[str]:
    candidates = {normalized}
    if normalized.startswith("BTC/"):
        candidates.add("XBT/" + normalized.split("/", 1)[1])
    if normalized.startswith("BTC") and "/" not in normalized:
        candidates.add("XBT" + normalized[3:])
    if normalized.startswith("XBT/"):
        candidates.add("BTC/" + normalized.split("/", 1)[1])
    if normalized.startswith("XBT") and "/" not in normalized:
        candidates.add("BTC" + normalized[3:])
    return {candidate for candidate in candidates if candidate}


def _fmt_num(value: float) -> str:
    return f"{float(value):.12f}".rstrip("0").rstrip(".")


def _raise_if_error(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or "error" not in payload:
        raise KrakenAPIError(f"Malformed response: {payload!r}")
    errors = payload.get("error") or []
    fatal = [error for error in errors if isinstance(error, str) and error.startswith("E")]
    if fatal:
        raise KrakenAPIError(f"Kraken error(s): {fatal} | full={errors}")
