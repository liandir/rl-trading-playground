from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


@dataclass(slots=True)
class SourceSpec:
    """Declarative description of a news source.

    ``kind`` selects the implementation in :mod:`news.sources` (e.g.
    ``"rss"``, ``"cryptopanic"``, ``"newsapi"``, ``"reddit"``,
    ``"twitter"``). ``credibility`` is a scalar in ``[0, 1]`` written
    onto every article produced by this source. ``options`` is passed
    through to the source constructor.
    """

    name: str
    kind: str
    credibility: float = 0.5
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NewsConfig:
    data_dir: Path
    aliases: dict[str, list[str]]
    sources: list[SourceSpec]
    relevance_threshold: float = 0.0
    case_sensitive_aliases: tuple[str, ...] = ()
    keep_raw: bool = True
    near_dup_hamming: int = 4

    @property
    def asset_universe(self) -> list[str]:
        return list(self.aliases.keys())

    def enabled_sources(self) -> list[SourceSpec]:
        return [s for s in self.sources if s.enabled]


def _load_aliases(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text())
    return {sym.upper(): list(a) for sym, a in raw.items()}


def load_config(path: str | os.PathLike[str]) -> NewsConfig:
    """Load a :class:`NewsConfig` from a TOML file.

    Layout::

        data_dir = "data/news"
        aliases_file = "configs/news_aliases.json"
        relevance_threshold = 0.0
        case_sensitive_aliases = ["NEAR"]
        keep_raw = true
        near_dup_hamming = 4

        [[sources]]
        name = "cryptopanic"
        kind = "cryptopanic"
        credibility = 0.6
        options = { api_key_env = "CRYPTOPANIC_API_KEY", filter = "hot" }
    """

    if tomllib is None:  # pragma: no cover
        raise RuntimeError("tomllib required (Python >= 3.11)")
    p = Path(path)
    raw = tomllib.loads(p.read_text())

    base = p.parent
    data_dir = Path(raw["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (base / data_dir).resolve()

    aliases_file = Path(raw["aliases_file"])
    if not aliases_file.is_absolute():
        aliases_file = (base / aliases_file).resolve()

    sources = [
        SourceSpec(
            name=s["name"],
            kind=s["kind"],
            credibility=float(s.get("credibility", 0.5)),
            enabled=bool(s.get("enabled", True)),
            options=dict(s.get("options", {})),
        )
        for s in raw.get("sources", [])
    ]

    return NewsConfig(
        data_dir=data_dir,
        aliases=_load_aliases(aliases_file),
        sources=sources,
        relevance_threshold=float(raw.get("relevance_threshold", 0.0)),
        case_sensitive_aliases=tuple(raw.get("case_sensitive_aliases", [])),
        keep_raw=bool(raw.get("keep_raw", True)),
        near_dup_hamming=int(raw.get("near_dup_hamming", 4)),
    )
