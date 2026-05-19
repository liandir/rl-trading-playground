"""News collection and storage for multi-asset trading.

Self-contained package. Designed for later extraction into its own
repository/uv-installable package: it does not import from any other
top-level module in this project, and all asset-specific configuration
(aliases, source allowlists, credibility weights) is injected from
outside via :class:`NewsConfig`.
"""

from __future__ import annotations

from .collector import Collector
from .config import NewsConfig, SourceSpec, load_config
from .models import Article, AssetMention, DedupHashes, RawArticle
from .storage import JsonlStore

__all__ = [
    "Article",
    "AssetMention",
    "Collector",
    "DedupHashes",
    "JsonlStore",
    "NewsConfig",
    "RawArticle",
    "SourceSpec",
    "load_config",
]
