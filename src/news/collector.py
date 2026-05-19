from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import NewsConfig
from .dedup import article_id, canonicalize_url, hashes_for, hamming64
from .models import Article, RawArticle
from .sources import build_source
from .storage import JsonlStore
from .tagging import AliasTagger

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CollectionStats:
    fetched: int = 0
    written: int = 0
    skipped_existing: int = 0
    skipped_no_assets: int = 0
    skipped_near_dup: int = 0
    errors: int = 0


class Collector:
    """Orchestrates fetch → tag → dedup → write.

    Constructed from a :class:`NewsConfig`; instantiates the configured
    sources via the lazy factory in :mod:`news.sources`. Near-dup
    detection compares a candidate article's 64-bit SimHash against the
    SimHashes of articles already stored for the same UTC day, across
    all sources, accepting if the minimum Hamming distance is greater
    than ``config.near_dup_hamming``.
    """

    def __init__(self, config: NewsConfig, store: JsonlStore | None = None) -> None:
        self.config = config
        self.store = store or JsonlStore(config.data_dir)
        self.tagger = AliasTagger(
            aliases=config.aliases,
            case_sensitive=config.case_sensitive_aliases,
            threshold=config.relevance_threshold,
        )

    def collect(
        self,
        since: datetime,
        until: datetime,
        *,
        source_names: list[str] | None = None,
    ) -> CollectionStats:
        since_u = since.astimezone(timezone.utc)
        until_u = until.astimezone(timezone.utc)
        stats = CollectionStats()

        specs = [s for s in self.config.enabled_sources()
                 if source_names is None or s.name in source_names]

        for spec in specs:
            try:
                src = build_source(spec, self.config.asset_universe)
            except Exception as e:
                log.exception("failed to build source %s: %s", spec.name, e)
                stats.errors += 1
                continue
            src.load_state(self.store.get_state(spec.name))
            try:
                for raw in src.fetch(since_u, until_u):
                    stats.fetched += 1
                    try:
                        if self._ingest(raw, stats):
                            stats.written += 1
                    except Exception:
                        log.exception("ingest failed for %s", raw.url)
                        stats.errors += 1
            except Exception:
                log.exception("source %s failed mid-fetch", spec.name)
                stats.errors += 1
            finally:
                state = src.dump_state()
                if state:
                    self.store.set_state(spec.name, state)

        return stats

    def _ingest(self, raw: RawArticle, stats: CollectionStats) -> bool:
        canon = canonicalize_url(raw.url)
        aid = article_id(canon)
        if self.store.has(aid):
            stats.skipped_existing += 1
            return False

        assets = self.tagger.tag(raw.title, raw.body, pre_tagged=raw.pre_tagged_assets)
        if not assets:
            stats.skipped_no_assets += 1
            return False

        dedup = hashes_for(raw.title, raw.body)
        dup_of = self._find_near_dup(raw.published_at, dedup.shingle_hash)
        if dup_of and dup_of == aid:
            dup_of = None

        published_at = raw.published_at.astimezone(timezone.utc)
        fetched_at = raw.fetched_at.astimezone(timezone.utc)
        latency = max(0.0, (fetched_at - published_at).total_seconds())

        raw_ref: str | None = None
        if self.config.keep_raw and raw.raw is not None:
            offset = self.store.write_raw(raw.source, published_at, {
                "id": aid,
                "url": raw.url,
                "external_id": raw.external_id,
                "payload": raw.raw,
            })
            raw_ref = f"raw/{raw.source}/{published_at.strftime('%Y-%m-%d')}.jsonl@{offset}"

        source_credibility = self._credibility_for(raw.source)

        article = Article(
            id=aid,
            source=raw.source,
            source_credibility=source_credibility,
            url=raw.url,
            canonical_url=canon,
            published_at=published_at.isoformat(timespec="seconds"),
            fetched_at=fetched_at.isoformat(timespec="seconds"),
            ingestion_latency_s=round(latency, 3),
            title=raw.title,
            body=raw.body,
            language=raw.language,
            assets=assets,
            dedup=dedup,
            duplicate_of=dup_of,
            raw_ref=raw_ref,
            external_id=raw.external_id,
        )
        if dup_of:
            stats.skipped_near_dup += 1
        return self.store.write(article)

    def _credibility_for(self, source_name: str) -> float:
        for s in self.config.sources:
            if s.name == source_name:
                return float(s.credibility)
        return 0.5

    def _find_near_dup(self, ts: datetime, shingle_hex: str) -> str | None:
        """Search the same UTC day across all sources for a near-dup."""
        candidate = int(shingle_hex, 16)
        from datetime import timedelta
        since = ts - timedelta(hours=12)
        until = ts + timedelta(hours=12)
        threshold = self.config.near_dup_hamming
        best_id: str | None = None
        best_dist = threshold + 1
        for other in self.store.iter_articles(since, until):
            try:
                other_hash = int(other.dedup.shingle_hash, 16)
            except (ValueError, AttributeError):
                continue
            dist = hamming64(candidate, other_hash)
            if dist <= threshold and dist < best_dist:
                best_id = other.id
                best_dist = dist
        return best_id
