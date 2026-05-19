"""Models utilities for news collection, tagging, deduplication, and storage utilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    """Utcnow for news collection, tagging, deduplication, and storage utilities.

    Returns:
        datetime: The computed or requested result.
    """
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class AssetMention:
    """AssetMention implementation for news collection, tagging, deduplication, and storage utilities."""
    symbol: str
    score: float
    mentions: int = 0


@dataclass(slots=True)
class DedupHashes:
    """DedupHashes implementation for news collection, tagging, deduplication, and storage utilities."""
    title_hash: str
    shingle_hash: str


@dataclass(slots=True)
class RawArticle:
    """Source-normalized record produced by a :class:`NewsSource`.

    A ``RawArticle`` is what every source must return. The collector
    enriches it (tagging, dedup, credibility) and writes the final
    :class:`Article`.
    """

    source: str
    url: str
    published_at: datetime
    title: str
    body: str | None = None
    language: str | None = None
    fetched_at: datetime = field(default_factory=_utcnow)
    external_id: str | None = None
    raw: dict[str, Any] | None = None
    pre_tagged_assets: list[str] | None = None


@dataclass(slots=True)
class Article:
    """Article implementation for news collection, tagging, deduplication, and storage utilities."""
    id: str
    source: str
    source_credibility: float
    url: str
    canonical_url: str
    published_at: str
    fetched_at: str
    ingestion_latency_s: float
    title: str
    body: str | None
    language: str | None
    assets: list[AssetMention]
    dedup: DedupHashes
    tags: list[str] = field(default_factory=list)
    duplicate_of: str | None = None
    sentiment: float | None = None
    event_type: str | None = None
    embedding: list[float] | None = None
    raw_ref: str | None = None
    external_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this object to a dictionary.

        Returns:
            dict[str, Any]: The computed or requested result.
        """
        d = asdict(self)
        d["assets"] = [asdict(m) for m in self.assets]
        d["dedup"] = asdict(self.dedup)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Article:
        """Build an instance from a dictionary.

        Args:
            d (dict[str, Any]): The d value.

        Returns:
            Article: The computed or requested result.
        """
        assets = [AssetMention(**m) for m in d.get("assets", [])]
        dedup = DedupHashes(**d["dedup"])
        kwargs = {**d, "assets": assets, "dedup": dedup}
        return cls(**kwargs)
