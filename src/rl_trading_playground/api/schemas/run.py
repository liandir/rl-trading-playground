"""Run record and status types."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from rl_trading_playground.api.schemas.agent import AgentConfig
from rl_trading_playground.api.schemas.data import DataConfig
from rl_trading_playground.api.schemas.env import EnvironmentConfig
from rl_trading_playground.api.schemas.training import TrainingConfig
from rl_trading_playground.api.schemas.validation import ValidationConfig


RunKind = Literal["training", "validation"]
RunStatus = Literal["queued", "running", "complete", "stopped", "failed"]


class RunSpec(BaseModel):
    """Frozen snapshot of every config used to launch a run."""

    kind: RunKind
    data: DataConfig
    env: EnvironmentConfig
    agent: AgentConfig
    training: TrainingConfig | None = None
    validation: ValidationConfig | None = None
    agent_id: str | None = Field(None, description="Existing agent id to load weights from.")
    env_id: str | None = None
    parent_run_id: str | None = None


class RunRecord(BaseModel):
    """A row in the runs table."""

    id: str
    kind: RunKind
    status: RunStatus
    name: str
    started_at: datetime
    ended_at: datetime | None = None
    agent_id: str | None = None
    env_id: str | None = None
    parent_run_id: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    notes: str = ""
    spec: RunSpec | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class CreateRunRequest(BaseModel):
    """Request body for POST /runs."""

    name: str | None = None
    spec: RunSpec
