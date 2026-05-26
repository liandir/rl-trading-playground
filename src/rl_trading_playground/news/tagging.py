"""Tagging utilities for news collection, tagging, deduplication, and storage utilities."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .models import AssetMention


@dataclass(slots=True)
class _CompiledAlias:
    """_CompiledAlias implementation for news collection, tagging, deduplication, and storage utilities."""
    symbol: str
    pattern: re.Pattern[str]


class AliasTagger:
    """Dictionary-based asset tagger.

    For each ``symbol`` we compile a regex covering its aliases as
    whole-word matches. Counts per symbol become a score in ``[0, 1]``
    via a saturating log scale. Multi-asset articles preserve all
    matches above ``threshold``.
    """

    def __init__(
        self,
        aliases: dict[str, list[str]],
        case_sensitive: tuple[str, ...] = (),
        threshold: float = 0.0,
        saturation: int = 6,
    ) -> None:
        """Initialize the instance.

        Args:
            aliases (dict[str, list[str]]): The aliases value.
            case_sensitive (tuple[str, ...]): The case sensitive value. Defaults to ``()``.
            threshold (float): The threshold value. Defaults to ``0.0``.
            saturation (int): The saturation value. Defaults to ``6``.

        Returns:
            None: This function does not return a value.
        """
        self._compiled: list[_CompiledAlias] = []
        cs = {s.upper() for s in case_sensitive}
        for sym, words in aliases.items():
            sym_u = sym.upper()
            escaped = sorted({re.escape(w) for w in words}, key=len, reverse=True)
            if not escaped:
                continue
            flags = 0 if sym_u in cs else re.IGNORECASE
            pat = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(escaped) + r")(?![A-Za-z0-9])", flags)
            self._compiled.append(_CompiledAlias(sym_u, pat))
        self._threshold = float(threshold)
        self._saturation = max(1, int(saturation))

    def tag(
        self,
        title: str,
        body: str | None = None,
        pre_tagged: list[str] | None = None,
    ) -> list[AssetMention]:
        """Tag for AliasTagger.

        Args:
            title (str): The title value.
            body (str | None): The body value. Defaults to ``None``.
            pre_tagged (list[str] | None): The pre tagged value. Defaults to ``None``.

        Returns:
            list[AssetMention]: The computed or requested result.
        """
        haystack = title if not body else f"{title}\n{body}"
        counts: dict[str, int] = {}
        for c in self._compiled:
            n = len(c.pattern.findall(haystack))
            if n:
                counts[c.symbol] = n

        # Source-provided tags get a small bias so pre-tagged but
        # alias-miss articles (e.g. CryptoPanic with a ticker-only
        # mention) still surface.
        if pre_tagged:
            for sym in pre_tagged:
                s = sym.upper()
                counts[s] = counts.get(s, 0) + 1

        if not counts:
            return []

        sat = self._saturation
        out: list[AssetMention] = []
        for sym, n in counts.items():
            score = math.log1p(n) / math.log1p(sat)
            score = min(score, 1.0)
            if score >= self._threshold:
                out.append(AssetMention(symbol=sym, score=round(score, 4), mentions=n))
        out.sort(key=lambda m: m.score, reverse=True)
        return out
