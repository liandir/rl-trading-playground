"""Validation rollout configuration."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ValidationConfig(BaseModel):
    """Options for a validation rollout."""

    start_index: int | None = Field(None, title="Start index", description="Defaults to 90% of data.")
    length: int = Field(10_000, ge=1, title="Length (steps)")
    explore: bool = Field(False, title="Exploratory")
    seed: int | None = Field(None, title="Seed")
