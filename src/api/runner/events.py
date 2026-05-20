"""JSONL event sink used by the runner subprocess."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO


class EventSink:
    """Append-only JSONL writer with one line per event.

    Each line is a self-contained JSON object matching
    :class:`src.api.schemas.event.RunEvent`. The file is flushed after every
    write so the parent FastAPI process can tail it in real time.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("a", buffering=1)

    def emit(
        self,
        kind: str,
        *,
        episode: int = 0,
        step: int = 0,
        percent: float = 0.0,
        avg_reward: float | None = None,
        avg_portfolio: float | None = None,
        sim_elapsed: str = "",
        metrics: dict[str, float] | None = None,
        message: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "t": time.time(),
            "kind": kind,
            "episode": episode,
            "step": step,
            "percent": float(percent),
            "avg_reward": avg_reward,
            "avg_portfolio": avg_portfolio,
            "sim_elapsed": sim_elapsed,
            "metrics": metrics or {},
            "message": message,
            "extra": extra or {},
        }
        self._fh.write(json.dumps(payload, default=_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "EventSink":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)
