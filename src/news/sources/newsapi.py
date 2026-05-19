from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timezone

from ..models import RawArticle
from .base import NewsSource
from ._http import get_json


class NewsApiSource(NewsSource):
    """newsapi.org ``/v2/everything`` source.

    Free tier limitation: only the last ~30 days are queryable. For
    older backfills use the paid plan. We issue one query per asset
    using its alias list to bias relevance.
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(
        self,
        *,
        name: str,
        credibility: float,
        asset_universe: list[str],
        aliases: dict[str, list[str]] | None = None,
        api_key_env: str = "NEWSAPI_KEY",
        language: str = "en",
        page_size: int = 100,
        max_pages: int = 5,
        sort_by: str = "publishedAt",
    ) -> None:
        super().__init__(name=name, credibility=credibility, asset_universe=asset_universe)
        self._api_key = os.getenv(api_key_env, "")
        self._aliases = aliases or {}
        self._language = language
        self._page_size = int(page_size)
        self._max_pages = int(max_pages)
        self._sort_by = sort_by

    def fetch(self, since: datetime, until: datetime) -> Iterator[RawArticle]:
        if not self._api_key:
            raise RuntimeError("NEWSAPI_KEY is not set")

        since_u = since.astimezone(timezone.utc)
        until_u = until.astimezone(timezone.utc)

        for symbol in self.asset_universe:
            aliases = self._aliases.get(symbol.upper(), [symbol])
            query = " OR ".join(f'"{a}"' for a in aliases[:6])  # API caps query length
            for page in range(1, self._max_pages + 1):
                payload = get_json(
                    self.BASE_URL,
                    params={
                        "q": query,
                        "from": since_u.isoformat(timespec="seconds"),
                        "to": until_u.isoformat(timespec="seconds"),
                        "language": self._language,
                        "pageSize": self._page_size,
                        "page": page,
                        "sortBy": self._sort_by,
                    },
                    headers={"X-Api-Key": self._api_key},
                )
                articles = payload.get("articles") or []
                if not articles:
                    break
                for art in articles:
                    ts = _parse_iso(art.get("publishedAt"))
                    if ts is None:
                        continue
                    yield RawArticle(
                        source=self.name,
                        url=art.get("url") or "",
                        published_at=ts,
                        title=(art.get("title") or "").strip(),
                        body=art.get("content") or art.get("description"),
                        language=self._language,
                        external_id=None,
                        pre_tagged_assets=[symbol.upper()],
                        raw=art,
                    )
                if len(articles) < self._page_size:
                    break


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None
