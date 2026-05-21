"""Deployment (live paper-trading session) schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.api.schemas.env import EnvironmentConfig


DeploymentStatus = Literal["queued", "running", "complete", "stopped", "failed"]
DeploymentMode = Literal["paper"]


class DeploymentSpec(BaseModel):
    """Frozen snapshot of every config used to launch a deployment."""

    agent_id: str = Field(..., description="Agent to drive the deployment.")
    env: EnvironmentConfig
    pairs: list[str] = Field(..., description="Kraken pair symbols, in env-column order.")
    asset_names: list[str] | None = Field(
        default=None,
        description="Display names for the pairs (defaults to the symbols).",
    )
    interval_minutes: int = Field(5, ge=1, title="Bar interval (minutes)")
    mode: DeploymentMode = Field("paper", title="Trading mode")
    override_interval_mismatch: bool = Field(
        False,
        description="Set true to launch even when the new interval differs from the agent's training interval.",
    )


class DeploymentRecord(BaseModel):
    """A row in the deployments table."""

    id: str
    status: DeploymentStatus
    name: str
    started_at: datetime
    ended_at: datetime | None = None
    agent_id: str
    pid: int | None = None
    exit_code: int | None = None
    notes: str = ""
    spec: DeploymentSpec | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class CreateDeploymentRequest(BaseModel):
    name: str | None = None
    spec: DeploymentSpec
