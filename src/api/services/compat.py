"""Guards that catch obvious agent / new-run incompatibilities up front.

The runner happily steps an agent at any bar interval, but inference
distribution depends on what the network saw during training. These
helpers read the checkpoint sidecar and surface a structured warning
or a hard error before launch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.api.schemas.agent import AgentRecord
from src.api.schemas.checkpoint import CheckpointMeta


@dataclass
class IntervalCheck:
    """Result of an interval-compatibility check."""

    ok: bool
    training_interval: int | None
    new_interval: int
    message: str = ""

    @property
    def is_unknown(self) -> bool:
        return self.training_interval is None


def read_sidecar(checkpoint_path: str | Path | None) -> CheckpointMeta | None:
    """Parse the .meta.json sibling of a checkpoint, if it exists."""

    if not checkpoint_path:
        return None
    path = Path(checkpoint_path).with_suffix(".meta.json")
    if not path.exists():
        return None
    try:
        return CheckpointMeta.model_validate(json.loads(path.read_text()))
    except Exception:
        return None


def check_interval(agent: AgentRecord | None, new_interval: int) -> IntervalCheck:
    """Compare ``new_interval`` against the agent's training interval if known."""

    if agent is None:
        return IntervalCheck(ok=True, training_interval=None, new_interval=new_interval)
    meta = read_sidecar(agent.checkpoint_path)
    training = meta.data_config.interval if meta and meta.data_config else None
    if training is None:
        return IntervalCheck(
            ok=True,
            training_interval=None,
            new_interval=new_interval,
            message="agent has no recorded training interval — proceed at your own risk",
        )
    if training != new_interval:
        return IntervalCheck(
            ok=False,
            training_interval=training,
            new_interval=new_interval,
            message=(
                f"agent was trained on {training}-minute bars, but the new run requests "
                f"{new_interval}-minute bars"
            ),
        )
    return IntervalCheck(ok=True, training_interval=training, new_interval=new_interval)
