"""Checkpoint records and sidecar metadata."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.api.schemas.agent import AgentConfig


class CheckpointMeta(BaseModel):
    """Sidecar JSON written alongside every `.ptm` file.

    The agent class only persists network weights, so the runner stores the
    matching agent/network config and a snapshot of the metric that triggered
    the save next to the weights file.
    """

    agent_config: AgentConfig
    step: int
    episode: int = 0
    metric_name: str | None = None
    metric_value: float | None = None
    parent_run_id: str | None = None
    saved_at: datetime
    extra: dict[str, Any] = Field(default_factory=dict)


class CheckpointRecord(BaseModel):
    """An indexed checkpoint row."""

    id: str
    run_id: str | None
    agent_id: str | None
    step: int
    metric_name: str | None
    metric_value: float | None
    tag: str | None
    path: str
    created_at: datetime
