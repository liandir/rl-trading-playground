"""Small stdlib HTTP helper shared by JSON-API sources.

Kept dependency-free on purpose (mirrors the urllib usage in the
existing kraken module). Wraps retries and JSON decoding.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class HttpError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    retries: int = 3,
    backoff: float = 1.5,
    user_agent: str = "news-collector/0.1",
) -> Any:
    if params:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{query}"

    req_headers = {"User-Agent": user_agent, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=req_headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(backoff ** attempt)
                last_exc = e
                continue
            raise HttpError(e.code, e.reason) from e
        except urllib.error.URLError as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("unreachable")
