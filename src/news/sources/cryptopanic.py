"""Cryptopanic utilities for news collection, tagging, deduplication, and storage utilities."""
from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timezone

from ..models import RawArticle
from .base import NewsSource
from ._http import get_json


class CryptoPanicSource(NewsSource):
    """CryptoPanic public posts API.

    Free tier: rate-limited but pre-tagged per coin. We filter by
    ``currencies`` (CryptoPanic's symbol set) so the call returns
    pre-relevant posts. Articles still go through the local tagger to
    pick up additional mentions and normalize scores.
    """

    BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(
        self,
        *,
        name: str,
        credibility: float,
        asset_universe: list[str],
        api_key_env: str = "CRYPTOPANIC_API_KEY",
        filter: str | None = None,
        kind: str = "news",
        public: bool = True,
        currencies: list[str] | None = None,
        max_pages: int = 20,
    ) -> None:
        """Initialize the instance.

        Args:
            name (str): The name value.
            credibility (float): The credibility value.
            asset_universe (list[str]): The asset universe value.
            api_key_env (str): The api key env value. Defaults to ``'CRYPTOPANIC_API_KEY'``.
            filter (str | None): The filter value. Defaults to ``None``.
            kind (str): The kind value. Defaults to ``'news'``.
            public (bool): The public value. Defaults to ``True``.
            currencies (list[str] | None): The currencies value. Defaults to ``None``.
            max_pages (int): The max pages value. Defaults to ``20``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name=name, credibility=credibility, asset_universe=asset_universe)
        self._api_key = os.getenv(api_key_env, "")
        self._filter = filter
        self._kind = kind
        self._public = public
        self._currencies = [c.upper() for c in (currencies or asset_universe)]
        self._max_pages = int(max_pages)

    def fetch(self, since: datetime, until: datetime) -> Iterator[RawArticle]:
        """Fetch items from the configured source.

        Args:
            since (datetime): The since value.
            until (datetime): The until value.

        Returns:
            Iterator[RawArticle]: The computed or requested result.
        """
        if not self._api_key:
            raise RuntimeError("CRYPTOPANIC_API_KEY is not set")

        since_u = since.astimezone(timezone.utc)
        until_u = until.astimezone(timezone.utc)

        url: str | None = self.BASE_URL
        params = {
            "auth_token": self._api_key,
            "kind": self._kind,
            "filter": self._filter,
            "public": "true" if self._public else None,
            "currencies": ",".join(self._currencies) if self._currencies else None,
        }

        pages = 0
        while url and pages < self._max_pages:
            payload = get_json(url, params=params if pages == 0 else None)
            for post in payload.get("results", []):
                ts = _parse_iso(post.get("published_at") or post.get("created_at"))
                if ts is None:
                    continue
                if ts < since_u:
                    return  # results are descending in time
                if ts > until_u:
                    continue
                title = (post.get("title") or "").strip()
                if not title:
                    continue
                src_url = post.get("url") or (post.get("source") or {}).get("domain")
                if not src_url:
                    continue
                pre = [c.get("code", "").upper() for c in (post.get("currencies") or [])]
                yield RawArticle(
                    source=self.name,
                    url=src_url,
                    published_at=ts,
                    title=title,
                    body=None,
                    language="en",
                    external_id=str(post.get("id")),
                    pre_tagged_assets=[c for c in pre if c],
                    raw=post,
                )
            url = payload.get("next")
            pages += 1


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse the iso.

    Args:
        ts (str | None): The ts value.

    Returns:
        datetime | None: The computed or requested result.
    """
    if not ts:
        return None
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None
