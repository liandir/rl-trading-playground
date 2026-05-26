"""Streaming run events written by the runner subprocess."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


EventKind = Literal[
    "run_started",
    "episode_start",
    "warm_up",
    "update",
    "episode_end",
    "checkpoint_saved",
    "validation_step",
    "run_stopped",
    "run_finished",
    "run_failed",
]


class RunEvent(BaseModel):
    """One line of `events.jsonl`."""

    t: float = Field(..., description="Wall-clock seconds since epoch.")
    kind: EventKind
    episode: int = 0
    step: int = 0
    percent: float = 0.0
    avg_reward: float | None = None
    avg_portfolio: float | None = None
    sim_elapsed: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    message: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)
