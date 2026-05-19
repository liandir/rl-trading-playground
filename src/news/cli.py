"""Cli utilities for news collection, tagging, deduplication, and storage utilities."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

from .collector import Collector
from .config import load_config
from .storage import JsonlStore


def _parse_when(value: str) -> datetime:
    """Accept ISO-8601 (``2026-05-17T08:00:00Z``) or ``-Nd`` / ``-Nh``.

    Args:
        value (str): The value value.

    Returns:
        datetime: The computed or requested result.
    """
    v = value.strip()
    if v.startswith("-") and v[-1] in ("d", "h", "m"):
        n = int(v[1:-1])
        delta = {
            "d": timedelta(days=n),
            "h": timedelta(hours=n),
            "m": timedelta(minutes=n),
        }[v[-1]]
        return datetime.now(timezone.utc) - delta
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _cmd_collect(args: argparse.Namespace) -> int:
    """Cmd collect for news collection, tagging, deduplication, and storage utilities.

    Args:
        args (argparse.Namespace): The args value.

    Returns:
        int: The computed or requested result.
    """
    cfg = load_config(args.config)
    coll = Collector(cfg)
    since = _parse_when(args.since)
    until = _parse_when(args.until) if args.until else datetime.now(timezone.utc)
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    stats = coll.collect(since, until, source_names=sources)
    print(json.dumps(stats.__dict__, indent=2))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """Cmd stats for news collection, tagging, deduplication, and storage utilities.

    Args:
        args (argparse.Namespace): The args value.

    Returns:
        int: The computed or requested result.
    """
    cfg = load_config(args.config)
    store = JsonlStore(cfg.data_dir)
    since = _parse_when(args.since)
    until = _parse_when(args.until) if args.until else datetime.now(timezone.utc)
    by_source: Counter[str] = Counter()
    by_asset: Counter[str] = Counter()
    n = 0
    for a in store.iter_articles(since, until):
        n += 1
        by_source[a.source] += 1
        for m in a.assets:
            by_asset[m.symbol] += 1
    print(json.dumps({
        "total": n,
        "by_source": dict(by_source.most_common()),
        "by_asset": dict(by_asset.most_common()),
        "range": [since.isoformat(), until.isoformat()],
    }, indent=2))
    return 0


def _cmd_reindex(args: argparse.Namespace) -> int:
    """Cmd reindex for news collection, tagging, deduplication, and storage utilities.

    Args:
        args (argparse.Namespace): The args value.

    Returns:
        int: The computed or requested result.
    """
    cfg = load_config(args.config)
    store = JsonlStore(cfg.data_dir)
    n = store.rebuild_index()
    print(json.dumps({"indexed": n}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the parser.

    Returns:
        argparse.ArgumentParser: The computed or requested result.
    """
    p = argparse.ArgumentParser(prog="news", description="Collect and inspect news articles")
    p.add_argument("--config", required=True, help="path to news config TOML")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="fetch and store articles")
    c.add_argument("--since", required=True, help="ISO timestamp or -Nd/-Nh/-Nm")
    c.add_argument("--until", default=None, help="ISO timestamp; defaults to now")
    c.add_argument("--sources", default=None, help="comma-separated source names to run")
    c.set_defaults(func=_cmd_collect)

    s = sub.add_parser("stats", help="summarize stored articles by source/asset")
    s.add_argument("--since", required=True)
    s.add_argument("--until", default=None)
    s.set_defaults(func=_cmd_stats)

    r = sub.add_parser("reindex", help="rebuild the by-id dedup index")
    r.set_defaults(func=_cmd_reindex)

    return p


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point.

    Args:
        argv (list[str] | None): The argv value. Defaults to ``None``.

    Returns:
        int: The computed or requested result.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
