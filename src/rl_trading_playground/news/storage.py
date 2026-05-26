"""Storage utilities for news collection, tagging, deduplication, and storage utilities."""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .models import Article


class JsonlStore:
    """Partitioned newline-delimited JSON store.

    Layout::

        <root>/articles/<source>/<YYYY-MM-DD>.jsonl
        <root>/raw/<source>/<YYYY-MM-DD>.jsonl
        <root>/state/<source>.json
        <root>/index/by_id.jsonl

    Partition key is the UTC date of ``published_at``. The ``by_id``
    index is append-only and used to skip articles that were already
    written. Rebuild it with :meth:`rebuild_index` if it drifts.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Initialize the instance.

        Args:
            root (str | os.PathLike[str]): The root value.

        Returns:
            None: This function does not return a value.
        """
        self.root = Path(root)
        (self.root / "articles").mkdir(parents=True, exist_ok=True)
        (self.root / "raw").mkdir(parents=True, exist_ok=True)
        (self.root / "state").mkdir(parents=True, exist_ok=True)
        (self.root / "index").mkdir(parents=True, exist_ok=True)
        self._known_ids: set[str] | None = None

    # ---- ids ----------------------------------------------------------------

    def _load_index(self) -> set[str]:
        """Load the index.

        Returns:
            set[str]: The computed or requested result.
        """
        if self._known_ids is not None:
            return self._known_ids
        ids: set[str] = set()
        idx = self.root / "index" / "by_id.jsonl"
        if idx.exists():
            with idx.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        ids.add(line)
        self._known_ids = ids
        return ids

    def has(self, article_id: str) -> bool:
        """Has for JsonlStore.

        Args:
            article_id (str): The article id value.

        Returns:
            bool: The computed or requested result.
        """
        return article_id in self._load_index()

    # ---- writes -------------------------------------------------------------

    def _articles_path(self, source: str, ts: datetime) -> Path:
        """Articles path for JsonlStore.

        Args:
            source (str): The source value.
            ts (datetime): The ts value.

        Returns:
            Path: The computed or requested result.
        """
        day = ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
        p = self.root / "articles" / source / f"{day}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _raw_path(self, source: str, ts: datetime) -> Path:
        """Raw path for JsonlStore.

        Args:
            source (str): The source value.
            ts (datetime): The ts value.

        Returns:
            Path: The computed or requested result.
        """
        day = ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
        p = self.root / "raw" / source / f"{day}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write(self, article: Article) -> bool:
        """Append an article. Returns ``False`` if already present.

        Args:
            article (Article): The article value.

        Returns:
            bool: The computed or requested result.
        """

        if self.has(article.id):
            return False
        path = self._articles_path(article.source, _parse_iso(article.published_at))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")
        with (self.root / "index" / "by_id.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(article.id + "\n")
        assert self._known_ids is not None
        self._known_ids.add(article.id)
        return True

    def write_raw(self, source: str, ts: datetime, payload: dict[str, Any]) -> int:
        """Write raw for JsonlStore.

        Args:
            source (str): The source value.
            ts (datetime): The ts value.
            payload (dict[str, Any]): The payload value.

        Returns:
            int: The computed or requested result.
        """
        path = self._raw_path(source, ts)
        with path.open("a", encoding="utf-8") as fh:
            offset = fh.tell()
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return offset

    # ---- state --------------------------------------------------------------

    def get_state(self, source: str) -> dict[str, Any]:
        """Return the current state snapshot.

        Args:
            source (str): The source value.

        Returns:
            dict[str, Any]: The computed or requested result.
        """
        p = self.root / "state" / f"{source}.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text())

    def set_state(self, source: str, state: dict[str, Any]) -> None:
        """Restore the current state snapshot.

        Args:
            source (str): The source value.
            state (dict[str, Any]): The state value.

        Returns:
            None: This function does not return a value.
        """
        p = self.root / "state" / f"{source}.json"
        fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp, p)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ---- reads --------------------------------------------------------------

    def iter_articles(
        self,
        since: datetime,
        until: datetime,
        *,
        sources: Iterable[str] | None = None,
        assets: Iterable[str] | None = None,
    ) -> Iterator[Article]:
        """Iterate over articles.

        Args:
            since (datetime): The since value.
            until (datetime): The until value.
            sources (Iterable[str] | None): The sources value. Defaults to ``None``.
            assets (Iterable[str] | None): The assets value. Defaults to ``None``.

        Returns:
            Iterator[Article]: The computed or requested result.
        """
        asset_filter = {a.upper() for a in assets} if assets else None
        src_dir = self.root / "articles"
        sources_to_scan: list[str]
        if sources is not None:
            sources_to_scan = list(sources)
        else:
            sources_to_scan = sorted(p.name for p in src_dir.iterdir() if p.is_dir()) if src_dir.exists() else []

        since_u = since.astimezone(timezone.utc)
        until_u = until.astimezone(timezone.utc)

        for src in sources_to_scan:
            for day_path in _iter_day_files(src_dir / src, since_u.date(), until_u.date()):
                with day_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        a = Article.from_dict(json.loads(line))
                        ts = _parse_iso(a.published_at)
                        if ts < since_u or ts > until_u:
                            continue
                        if asset_filter is not None and not any(m.symbol in asset_filter for m in a.assets):
                            continue
                        yield a

    # ---- maintenance --------------------------------------------------------

    def rebuild_index(self) -> int:
        """Rebuild index for JsonlStore.

        Returns:
            int: The computed or requested result.
        """
        ids: list[str] = []
        articles_dir = self.root / "articles"
        if articles_dir.exists():
            for jl in articles_dir.rglob("*.jsonl"):
                with jl.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ids.append(json.loads(line)["id"])
                        except (KeyError, json.JSONDecodeError):
                            continue
        idx = self.root / "index" / "by_id.jsonl"
        idx.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        self._known_ids = set(ids)
        return len(ids)


def _parse_iso(ts: str) -> datetime:
    """Parse the iso.

    Args:
        ts (str): The ts value.

    Returns:
        datetime: The computed or requested result.
    """
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _iter_day_files(root: Path, since: date, until: date) -> Iterator[Path]:
    """Iterate over day files.

    Args:
        root (Path): The root value.
        since (date): The since value.
        until (date): The until value.

    Returns:
        Iterator[Path]: The computed or requested result.
    """
    if not root.exists():
        return
    d = since
    while d <= until:
        p = root / f"{d.isoformat()}.jsonl"
        if p.exists():
            yield p
        d += timedelta(days=1)
