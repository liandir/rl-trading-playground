"""Rss utilities for news collection, tagging, deduplication, and storage utilities."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from ..models import RawArticle
from .base import NewsSource


class RssSource(NewsSource):
    """Generic RSS / Atom source.

    One :class:`RssSource` can wrap several feed URLs that share the
    same credibility (e.g. all CoinDesk feeds). Body extraction is
    summary-only by default; if ``fetch_body`` is true and
    ``trafilatura`` is installed, the article HTML is downloaded and
    the main text is extracted.
    """

    def __init__(
        self,
        *,
        name: str,
        credibility: float,
        asset_universe: list[str],
        feeds: list[str],
        language: str | None = "en",
        fetch_body: bool = False,
    ) -> None:
        """Initialize the instance.

        Args:
            name (str): The name value.
            credibility (float): The credibility value.
            asset_universe (list[str]): The asset universe value.
            feeds (list[str]): The feeds value.
            language (str | None): The language value. Defaults to ``'en'``.
            fetch_body (bool): The fetch body value. Defaults to ``False``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name=name, credibility=credibility, asset_universe=asset_universe)
        self.feeds = list(feeds)
        self.language = language
        self.fetch_body = bool(fetch_body)

    def fetch(self, since: datetime, until: datetime) -> Iterator[RawArticle]:
        """Fetch items from the configured source.

        Args:
            since (datetime): The since value.
            until (datetime): The until value.

        Returns:
            Iterator[RawArticle]: The computed or requested result.
        """
        try:
            import feedparser  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError(
                "RssSource requires 'feedparser'. Install with `uv add feedparser`."
            ) from e

        since_u = since.astimezone(timezone.utc)
        until_u = until.astimezone(timezone.utc)

        for feed_url in self.feeds:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries:
                ts = _entry_timestamp(entry)
                if ts is None or ts < since_u or ts > until_u:
                    continue
                title = (entry.get("title") or "").strip()
                url = (entry.get("link") or "").strip()
                if not url or not title:
                    continue
                body = entry.get("summary") or entry.get("description")
                if self.fetch_body:
                    body = _extract_body(url) or body
                yield RawArticle(
                    source=self.name,
                    url=url,
                    published_at=ts,
                    title=title,
                    body=body,
                    language=self.language,
                    external_id=entry.get("id"),
                    raw={"feed": feed_url, "entry": _entry_to_dict(entry)},
                )


def _entry_timestamp(entry: Any) -> datetime | None:
    """Entry timestamp for news collection, tagging, deduplication, and storage utilities.

    Args:
        entry (Any): The entry value.

    Returns:
        datetime | None: The computed or requested result.
    """
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    import calendar
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)


def _entry_to_dict(entry: Any) -> dict[str, Any]:
    """Entry to dict for news collection, tagging, deduplication, and storage utilities.

    Args:
        entry (Any): The entry value.

    Returns:
        dict[str, Any]: The computed or requested result.
    """
    out: dict[str, Any] = {}
    for k in ("title", "link", "id", "summary", "published", "updated", "author"):
        v = entry.get(k)
        if v is not None:
            out[k] = v
    return out


def _extract_body(url: str) -> str | None:
    """Extract body for news collection, tagging, deduplication, and storage utilities.

    Args:
        url (str): The url value.

    Returns:
        str | None: The computed or requested result.
    """
    try:
        import trafilatura  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        return trafilatura.extract(downloaded) or None
    except Exception:
        return None
