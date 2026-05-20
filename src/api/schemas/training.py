"""Training loop configuration."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Options for a training run executed by `src.api.runner`."""

    n_episodes: int = Field(3, ge=1, title="Episodes")
    batch_size: int = Field(8, ge=1, le=512, title="Batch size")
    max_steps: int = Field(50_000, ge=4, title="Max steps")
    warm_up: int = Field(10_000, ge=0, title="Warm-up steps")
    update_interval: int = Field(128, ge=2, title="Update interval")
    n_updates: int = Field(4, ge=1, title="Updates per rollout")
    burn_in_updates: int = Field(1, ge=0, title="Burn-in updates")
    lr: float = Field(1e-5, gt=0.0, title="Learning rate")
    optim: Literal["AdamW", "Adam"] = Field("AdamW", title="Optimizer")
    init_optimizer: bool = Field(False, title="Force optimizer init")
    max_grad_norm: float | None = Field(None, title="Max grad norm")
    save_checkpoint: bool = Field(True, title="Save checkpoint at end")
    seed: int | None = Field(None, title="Seed")
