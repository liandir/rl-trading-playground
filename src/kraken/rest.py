"""Rest utilities for Kraken market data and exchange integration helpers."""
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
    """KrakenAPIError exception type for Kraken market data and exchange integration helpers."""
    pass


class _Nonce:
    """_Nonce implementation for Kraken market data and exchange integration helpers."""
    def __init__(self) -> None:
        """Initialize the instance.

        Returns:
            None: This function does not return a value.
        """
        self._lock = threading.Lock()
        self._last = int(time.time() * 1e3)

    def next(self) -> str:
        """Next for _Nonce.

        Returns:
            str: The computed or requested result.
        """
        with self._lock:
            now = int(time.time() * 1e3)
            self._last = max(self._last + 1, now)
            return str(self._last)


class KrakenSpotClient:
    """KrakenSpotClient client for Kraken market data and exchange integration helpers."""
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        *,
        user_agent: str = "kraken-spot-python/2.0",
        timeout: float = 10.0,
        rate_limit_per_sec: float = 15.0,
    ) -> None:
        """Initialize the instance.

        Args:
            api_key (str | None): The api key value. Defaults to ``None``.
            api_secret (str | None): The api secret value. Defaults to ``None``.
            user_agent (str): The user agent value. Defaults to ``'kraken-spot-python/2.0'``.
            timeout (float): The timeout value. Defaults to ``10.0``.
            rate_limit_per_sec (float): The rate limit per sec value. Defaults to ``15.0``.

        Returns:
            None: This function does not return a value.
        """
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
        """Resolve the pair.

        Args:
            pair_like (str): The pair like value.

        Returns:
            str: The computed or requested result.
        """
        return self.resolve_ws_pair(pair_like)

    def resolve_ws_pair(self, pair_like: str) -> str:
        """Resolve the ws pair.

        Args:
            pair_like (str): The pair like value.

        Returns:
            str: The computed or requested result.
        """
        pair_code, meta = self._find_pair_meta(pair_like)
        return meta.get("wsname") or meta.get("altname") or pair_code

    def resolve_rest_pair(self, pair_like: str) -> str:
        """Resolve the rest pair.

        Args:
            pair_like (str): The pair like value.

        Returns:
            str: The computed or requested result.
        """
        pair_code, meta = self._find_pair_meta(pair_like)
        return meta.get("altname") or pair_code

    def get_pair_metadata(self, pair_like: str) -> dict[str, Any]:
        """Return the pair metadata.

        Args:
            pair_like (str): The pair like value.

        Returns:
            dict[str, Any]: The computed or requested result.
        """
        _, meta = self._find_pair_meta(pair_like)
        return dict(meta)

    def get_ticker(self, pair_like: str) -> dict[str, Any]:
        """Return the ticker.

        Args:
            pair_like (str): The pair like value.

        Returns:
            dict[str, Any]: The computed or requested result.
        """
        pair = self.resolve_rest_pair(pair_like)
        data = self._get_public("/0/public/Ticker", {"pair": pair})
        result = data["result"]
        first_key = next(iter(result.keys()))
        ticker = result[first_key]
        ticker["_pair"] = first_key
        return ticker

    def get_order_book(self, pair_like: str, depth: int = 10) -> dict[str, Any]:
        """Return the order book.

        Args:
            pair_like (str): The pair like value.
            depth (int): The depth value. Defaults to ``10``.

        Returns:
            dict[str, Any]: The computed or requested result.
        """
        pair = self.resolve_rest_pair(pair_like)
        depth = max(1, min(int(depth), 500))
        data = self._get_public("/0/public/Depth", {"pair": pair, "count": str(depth)})
        result = data["result"]
        first_key = next(iter(result.keys()))
        book = result[first_key]
        book["_pair"] = first_key
        return book

    def get_balance(self) -> dict[str, str]:
        """Return the balance.

        Returns:
            dict[str, str]: The computed or requested result.
        """
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
        """Place order for KrakenSpotClient.

        Args:
            pair_like (str): The pair like value.
            side (str): The side value.
            ordertype (str): The ordertype value. Defaults to ``'market'``.
            volume (float): The volume value. Defaults to ``0.0``.
            price (float | None): The price value. Defaults to ``None``.
            userref (int | None): The userref value. Defaults to ``None``.
            timeinforce (str | None): The timeinforce value. Defaults to ``None``.
            starttm (str | None): The starttm value. Defaults to ``None``.
            expiretm (str | None): The expiretm value. Defaults to ``None``.
            validate (bool): The validate value. Defaults to ``False``.
            oflags (str | None): The oflags value. Defaults to ``None``.
            leverage (str | None): The leverage value. Defaults to ``None``.
            trading_agreement (bool): The trading agreement value. Defaults to ``True``.

        Returns:
            dict[str, Any]: The computed or requested result.
        """
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
        """Cancel order for KrakenSpotClient.

        Args:
            txid (str): The txid value.

        Returns:
            dict[str, Any]: The computed or requested result.
        """
        return self._post_private("/0/private/CancelOrder", {"txid": txid})["result"]

    def query_orders_info(
        self,
        txid: str | None = None,
        *,
        userref: int | None = None,
        trades: bool = False,
        consolidate_taker: bool = False,
    ) -> dict[str, Any]:
        """Query orders info for KrakenSpotClient.

        Args:
            txid (str | None): The txid value. Defaults to ``None``.
            userref (int | None): The userref value. Defaults to ``None``.
            trades (bool): The trades value. Defaults to ``False``.
            consolidate_taker (bool): The consolidate taker value. Defaults to ``False``.

        Returns:
            dict[str, Any]: The computed or requested result.
        """
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
        """Find pair meta for KrakenSpotClient.

        Args:
            pair_like (str): The pair like value.

        Returns:
            tuple[str, dict[str, Any]]: The computed or requested result.
        """
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
        """Rate limit for KrakenSpotClient.

        Returns:
            None: This function does not return a value.
        """
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
        """Return the public.

        Args:
            path (str): The path value.
            params (dict[str, str] | None): The params value. Defaults to ``None``.

        Returns:
            dict[str, Any]: The computed or requested result.
        """
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
        """Post private for KrakenSpotClient.

        Args:
            path (str): The path value.
            params (dict[str, Any]): The params value.

        Returns:
            dict[str, Any]: The computed or requested result.
        """
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
    """Normalize pair like for Kraken market data and exchange integration helpers.

    Args:
        value (str): The value value.

    Returns:
        str: The computed or requested result.
    """
    return value.strip().upper().replace("-", "/").replace(" ", "")


def _pair_candidates(normalized: str) -> set[str]:
    """Pair candidates for Kraken market data and exchange integration helpers.

    Args:
        normalized (str): The normalized value.

    Returns:
        set[str]: The computed or requested result.
    """
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
    """Fmt num for Kraken market data and exchange integration helpers.

    Args:
        value (float): The value value.

    Returns:
        str: The computed or requested result.
    """
    return f"{float(value):.12f}".rstrip("0").rstrip(".")


def _raise_if_error(payload: dict[str, Any]) -> None:
    """Raise if error for Kraken market data and exchange integration helpers.

    Args:
        payload (dict[str, Any]): The payload value.

    Returns:
        None: This function does not return a value.
    """
    if not isinstance(payload, dict) or "error" not in payload:
        raise KrakenAPIError(f"Malformed response: {payload!r}")
    errors = payload.get("error") or []
    fatal = [error for error in errors if isinstance(error, str) and error.startswith("E")]
    if fatal:
        raise KrakenAPIError(f"Kraken error(s): {fatal} | full={errors}")
