"""Reddit utilities for news collection, tagging, deduplication, and storage utilities."""
from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timezone

from ..models import RawArticle
from .base import NewsSource


class RedditSource(NewsSource):
    """Reddit source via PRAW.

    Uses script-type OAuth credentials. Set:

    - ``REDDIT_CLIENT_ID``
    - ``REDDIT_CLIENT_SECRET``
    - ``REDDIT_USER_AGENT`` (defaults to ``"news-collector/0.1"``)

    ``mode='new'`` streams the ``new`` listing of each subreddit and
    stops once posts are older than ``since``. Reddit's listing API
    paginates ~1000 deep; for older backfills we'd need Pushshift /
    Arctic Shift, which is out of scope here.

    ``mode='search'`` runs a query per asset across the subreddits.
    Useful when subreddits are large and you only care about specific
    coins.
    """

    def __init__(
        self,
        *,
        name: str,
        credibility: float,
        asset_universe: list[str],
        subreddits: list[str],
        aliases: dict[str, list[str]] | None = None,
        mode: str = "new",
        limit: int = 500,
        include_comments_top_n: int = 0,
        client_id_env: str = "REDDIT_CLIENT_ID",
        client_secret_env: str = "REDDIT_CLIENT_SECRET",
        user_agent_env: str = "REDDIT_USER_AGENT",
    ) -> None:
        """Initialize the instance.

        Args:
            name (str): The name value.
            credibility (float): The credibility value.
            asset_universe (list[str]): The asset universe value.
            subreddits (list[str]): The subreddits value.
            aliases (dict[str, list[str]] | None): The aliases value. Defaults to ``None``.
            mode (str): The mode value. Defaults to ``'new'``.
            limit (int): The limit value. Defaults to ``500``.
            include_comments_top_n (int): The include comments top n value. Defaults to ``0``.
            client_id_env (str): The client id env value. Defaults to ``'REDDIT_CLIENT_ID'``.
            client_secret_env (str): The client secret env value. Defaults to ``'REDDIT_CLIENT_SECRET'``.
            user_agent_env (str): The user agent env value. Defaults to ``'REDDIT_USER_AGENT'``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name=name, credibility=credibility, asset_universe=asset_universe)
        self.subreddits = list(subreddits)
        self.aliases = aliases or {}
        self.mode = mode
        self.limit = int(limit)
        self.include_comments_top_n = int(include_comments_top_n)
        self._client_id = os.getenv(client_id_env, "")
        self._client_secret = os.getenv(client_secret_env, "")
        self._user_agent = os.getenv(user_agent_env, "news-collector/0.1")

    def _client(self):  # pragma: no cover - thin wrapper
        """Client for RedditSource.

        Returns:
            Any: The computed or requested result.
        """
        try:
            import praw  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "RedditSource requires 'praw'. Install with `uv add praw`."
            ) from e
        if not (self._client_id and self._client_secret):
            raise RuntimeError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set")
        return praw.Reddit(
            client_id=self._client_id,
            client_secret=self._client_secret,
            user_agent=self._user_agent,
            check_for_async=False,
        )

    def fetch(self, since: datetime, until: datetime) -> Iterator[RawArticle]:
        """Fetch items from the configured source.

        Args:
            since (datetime): The since value.
            until (datetime): The until value.

        Returns:
            Iterator[RawArticle]: The computed or requested result.
        """
        reddit = self._client()
        since_u = since.astimezone(timezone.utc)
        until_u = until.astimezone(timezone.utc)

        if self.mode == "new":
            for sub in self.subreddits:
                yield from self._iter_listing(reddit, sub, since_u, until_u)
        elif self.mode == "search":
            for symbol in self.asset_universe:
                aliases = self.aliases.get(symbol.upper(), [symbol])
                query = " OR ".join(f'"{a}"' for a in aliases[:6])
                for sub in self.subreddits:
                    yield from self._iter_search(reddit, sub, query, symbol, since_u, until_u)
        else:
            raise ValueError(f"unknown reddit mode: {self.mode!r}")

    def _post_to_raw(self, post, subreddit_name: str, pre_tagged: list[str] | None = None) -> RawArticle:
        """Post to raw for RedditSource.

        Args:
            post (Any): The post value.
            subreddit_name (str): The subreddit name value.
            pre_tagged (list[str] | None): The pre tagged value. Defaults to ``None``.

        Returns:
            RawArticle: The computed or requested result.
        """
        ts = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
        body = post.selftext or None
        if self.include_comments_top_n > 0:
            try:
                post.comments.replace_more(limit=0)
                top = sorted(post.comments.list(), key=lambda c: getattr(c, "score", 0), reverse=True)
                snippets = [c.body for c in top[: self.include_comments_top_n] if getattr(c, "body", None)]
                if snippets:
                    body = (body or "") + "\n\n--- comments ---\n" + "\n\n".join(snippets)
            except Exception:
                pass
        return RawArticle(
            source=self.name,
            url=f"https://www.reddit.com{post.permalink}",
            published_at=ts,
            title=post.title or "",
            body=body,
            language="en",
            external_id=post.id,
            pre_tagged_assets=pre_tagged,
            raw={"subreddit": subreddit_name, "score": post.score, "num_comments": post.num_comments},
        )

    def _iter_listing(self, reddit, sub: str, since_u: datetime, until_u: datetime) -> Iterator[RawArticle]:
        """Iterate over listing.

        Args:
            reddit (Any): The reddit value.
            sub (str): The sub value.
            since_u (datetime): The since u value.
            until_u (datetime): The until u value.

        Returns:
            Iterator[RawArticle]: The computed or requested result.
        """
        for post in reddit.subreddit(sub).new(limit=self.limit):
            ts = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            if ts < since_u:
                break
            if ts > until_u:
                continue
            yield self._post_to_raw(post, sub)

    def _iter_search(
        self,
        reddit,
        sub: str,
        query: str,
        symbol: str,
        since_u: datetime,
        until_u: datetime,
    ) -> Iterator[RawArticle]:
        """Iterate over search.

        Args:
            reddit (Any): The reddit value.
            sub (str): The sub value.
            query (str): The query value.
            symbol (str): The symbol value.
            since_u (datetime): The since u value.
            until_u (datetime): The until u value.

        Returns:
            Iterator[RawArticle]: The computed or requested result.
        """
        for post in reddit.subreddit(sub).search(query, sort="new", time_filter="all", limit=self.limit):
            ts = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            if ts < since_u or ts > until_u:
                continue
            yield self._post_to_raw(post, sub, pre_tagged=[symbol.upper()])
