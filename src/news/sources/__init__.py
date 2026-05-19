from __future__ import annotations

from typing import Any

from ..config import SourceSpec
from .base import NewsSource


def build_source(spec: SourceSpec, asset_universe: list[str]) -> NewsSource:
    """Factory: turn a :class:`SourceSpec` into a concrete source.

    Lazily imports the backend module so optional dependencies (praw,
    feedparser, tweepy, ...) are only required when actually enabled.
    """

    opts: dict[str, Any] = dict(spec.options)
    opts.setdefault("name", spec.name)
    opts.setdefault("credibility", spec.credibility)
    opts.setdefault("asset_universe", asset_universe)

    kind = spec.kind.lower()
    if kind == "rss":
        from .rss import RssSource
        return RssSource(**opts)
    if kind == "cryptopanic":
        from .cryptopanic import CryptoPanicSource
        return CryptoPanicSource(**opts)
    if kind == "newsapi":
        from .newsapi import NewsApiSource
        return NewsApiSource(**opts)
    if kind == "reddit":
        from .reddit import RedditSource
        return RedditSource(**opts)
    if kind == "twitter":
        from .twitter import TwitterSource
        return TwitterSource(**opts)
    raise ValueError(f"unknown source kind: {spec.kind!r}")


__all__ = ["NewsSource", "build_source"]
