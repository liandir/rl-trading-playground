from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime

from ..models import RawArticle


class NewsSource(ABC):
    """Abstract source contract.

    Implementations must yield :class:`RawArticle` instances whose
    ``published_at`` is timezone-aware UTC. The collector is responsible
    for tagging, dedup, and writing — sources should stay dumb.
    """

    name: str
    credibility: float

    def __init__(self, *, name: str, credibility: float, asset_universe: list[str]) -> None:
        self.name = name
        self.credibility = float(credibility)
        self.asset_universe = list(asset_universe)

    @abstractmethod
    def fetch(self, since: datetime, until: datetime) -> Iterator[RawArticle]:
        ...

    # Optional hook: per-source resumable cursor. The collector passes
    # the dict from :meth:`JsonlStore.get_state` in and stores whatever
    # the source returns. Default is no-op.
    def load_state(self, state: dict) -> None:  # pragma: no cover
        return None

    def dump_state(self) -> dict:  # pragma: no cover
        return {}
