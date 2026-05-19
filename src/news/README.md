# news

Self-contained news collection & storage package. Pluggable sources,
normalized JSONL store, asset tagging via alias dictionary.

This package is intentionally **independent** of the rest of this
repository (no imports from `kraken`, `agent`, `environment`, ...). It
is designed to be extracted into its own repo later and installed via
`uv` (the same way `forex` is).

## Quickstart

```bash
# 1. Drop a config + aliases file (see configs/news.toml.example)
# 2. Set whichever API keys you need
export CRYPTOPANIC_API_KEY=...
export NEWSAPI_KEY=...
export REDDIT_CLIENT_ID=...
export REDDIT_CLIENT_SECRET=...

# 3. Collect the last 24h
#    (use --since=-24h, not --since -24h — argparse parses bare -24h as a flag)
python -m news --config configs/news.toml collect --since=-24h

# 4. Inspect what's stored
python -m news --config configs/news.toml stats --since=-7d
```

## Layout

```
src/news/
    models.py          Article, RawArticle, AssetMention, DedupHashes
    config.py          NewsConfig (TOML), SourceSpec
    tagging.py         alias-dictionary asset tagger
    dedup.py           canonical URL, title hash, 64-bit SimHash
    storage.py         partitioned JSONL store + by-id index
    collector.py       orchestrator: fetch → tag → dedup → write
    sources/
        base.py        NewsSource ABC
        rss.py         generic RSS / Atom (CoinDesk, CoinTelegraph, ...)
        cryptopanic.py CryptoPanic API
        newsapi.py     newsapi.org /v2/everything
        reddit.py      PRAW-backed subreddit reader
        twitter.py     scaffold with tweepy_v2 / snscrape / lunarcrush
    cli.py             python -m news ...
```

## Storage format

```
<data_dir>/
    articles/<source>/<YYYY-MM-DD>.jsonl   normalized Article per line
    raw/<source>/<YYYY-MM-DD>.jsonl        untouched source payloads
    state/<source>.json                    per-source resume state
    index/by_id.jsonl                      append-only dedup index
```

Partition key is the UTC date of `published_at`. Range queries open
only the days they need.

## Programmatic use

```python
from datetime import datetime, timezone

from news import Collector, JsonlStore, load_config

cfg = load_config("configs/news.toml")
coll = Collector(cfg)
coll.collect(
    since=datetime(2026, 5, 1, tzinfo=timezone.utc),
    until=datetime(2026, 5, 17, tzinfo=timezone.utc),
)

store = JsonlStore(cfg.data_dir)
for a in store.iter_articles(
    since=datetime(2026, 5, 16, tzinfo=timezone.utc),
    until=datetime(2026, 5, 17, tzinfo=timezone.utc),
    assets=["BTC", "ETH"],
):
    print(a.published_at, a.source, a.title)
```

## Adding a source

1. Subclass `news.sources.base.NewsSource`, implement `fetch(since, until)`
   to yield `RawArticle`s.
2. Register it in the `kind` switch in `news/sources/__init__.py`.
3. Add a `[[sources]]` entry to your config TOML.

Sources should stay dumb: no tagging, no dedup, no IO to the store —
the collector handles all of that.

## Optional dependencies

- `feedparser` — RSS source
- `praw` — Reddit source
- `trafilatura` — full-body extraction for RSS (otherwise summary-only)
- `tweepy` — Twitter v2 backend (when wired)

These are imported lazily; you only need the ones for sources you
actually enable.

## Extraction to a standalone package

When ready:

1. Move `src/news/` into its own repo.
2. Add a `pyproject.toml` with the optional extras above.
3. In this repo, replace the in-tree directory with a `uv` dependency
   (`news = { git = "ssh://git@.../news.git" }`).

No call sites in this repo will need to change because no other module
imports from `news` yet.
