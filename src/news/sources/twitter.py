"""Twitter utilities for news collection, tagging, deduplication, and storage utilities."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from ..models import RawArticle
from .base import NewsSource


class TwitterSource(NewsSource):
    """X / Twitter source with a swappable backend.

    Backends:

    - ``tweepy_v2``: official v2 API via ``tweepy`` (requires paid plan,
      ``TWITTER_BEARER_TOKEN`` env var).
    - ``snscrape``:  unofficial scraper; brittle, ToS-grey, but free.
    - ``lunarcrush``: paid third-party aggregator with crypto-curated
      social feeds.

    All backends are scaffolded as :class:`NotImplementedError` for
    now — pick one and we'll wire it. The selection point is a single
    branch in :meth:`fetch`, so swapping backends later is cheap.
    """

    def __init__(
        self,
        *,
        name: str,
        credibility: float,
        asset_universe: list[str],
        backend: str = "tweepy_v2",
        accounts: list[str] | None = None,
        aliases: dict[str, list[str]] | None = None,
        cashtags: bool = True,
        max_results_per_query: int = 100,
    ) -> None:
        """Initialize the instance.

        Args:
            name (str): The name value.
            credibility (float): The credibility value.
            asset_universe (list[str]): The asset universe value.
            backend (str): The backend value. Defaults to ``'tweepy_v2'``.
            accounts (list[str] | None): The accounts value. Defaults to ``None``.
            aliases (dict[str, list[str]] | None): The aliases value. Defaults to ``None``.
            cashtags (bool): The cashtags value. Defaults to ``True``.
            max_results_per_query (int): The max results per query value. Defaults to ``100``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name=name, credibility=credibility, asset_universe=asset_universe)
        self.backend = backend
        self.accounts = list(accounts or [])
        self.aliases = aliases or {}
        self.cashtags = bool(cashtags)
        self.max_results_per_query = int(max_results_per_query)

    def fetch(self, since: datetime, until: datetime) -> Iterator[RawArticle]:
        """Fetch items from the configured source.

        Args:
            since (datetime): The since value.
            until (datetime): The until value.

        Returns:
            Iterator[RawArticle]: The computed or requested result.
        """
        if self.backend == "tweepy_v2":
            return self._fetch_tweepy(since, until)
        if self.backend == "snscrape":
            return self._fetch_snscrape(since, until)
        if self.backend == "lunarcrush":
            return self._fetch_lunarcrush(since, until)
        raise ValueError(f"unknown twitter backend: {self.backend!r}")

    def _build_queries(self) -> list[tuple[str, list[str]]]:
        """Build the queries.

        Returns:
            list[tuple[str, list[str]]]: The computed or requested result.
        """
        queries: list[tuple[str, list[str]]] = []
        for symbol in self.asset_universe:
            aliases = self.aliases.get(symbol.upper(), [symbol])
            terms = [f'"{a}"' for a in aliases[:5]]
            if self.cashtags:
                terms.append(f"${symbol}")
            queries.append((" OR ".join(terms), [symbol.upper()]))
        if self.accounts:
            queries.append(
                (" OR ".join(f"from:{a.lstrip('@')}" for a in self.accounts), [])
            )
        return queries

    # Concrete backends below are intentional stubs. Implement when the
    # credential/cost trade-off is decided.

    def _fetch_tweepy(self, since: datetime, until: datetime) -> Iterator[RawArticle]:  # pragma: no cover
        """Fetch tweepy for TwitterSource.

        Args:
            since (datetime): The since value.
            until (datetime): The until value.

        Returns:
            Iterator[RawArticle]: The computed or requested result.
        """
        raise NotImplementedError("tweepy_v2 backend not wired yet — needs TWITTER_BEARER_TOKEN")

    def _fetch_snscrape(self, since: datetime, until: datetime) -> Iterator[RawArticle]:  # pragma: no cover
        """Fetch snscrape for TwitterSource.

        Args:
            since (datetime): The since value.
            until (datetime): The until value.

        Returns:
            Iterator[RawArticle]: The computed or requested result.
        """
        raise NotImplementedError("snscrape backend not wired yet")

    def _fetch_lunarcrush(self, since: datetime, until: datetime) -> Iterator[RawArticle]:  # pragma: no cover
        """Fetch lunarcrush for TwitterSource.

        Args:
            since (datetime): The since value.
            until (datetime): The until value.

        Returns:
            Iterator[RawArticle]: The computed or requested result.
        """
        raise NotImplementedError("lunarcrush backend not wired yet — needs LUNARCRUSH_API_KEY")
