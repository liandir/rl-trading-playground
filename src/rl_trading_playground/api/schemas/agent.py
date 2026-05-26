"""Agent + network configuration schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


AgentType = Literal[
    "aac",
    "aaq",
    "ppo",
    "aac_aux",
    "aaq_aux",
    "ppo_aux",
    "aac_model",
    "spatiotemporal_aac",
]

NetworkPreset = Literal[
    "attention_memory",
    "flat_per_asset",
    "auxiliary_per_asset",
    "flat_model",
    "attention_model",
    "spatiotemporal_auxiliary",
    "tiny",
]


class AgentConfig(BaseModel):
    """Construction options for an agent and its network."""

    agent_type: AgentType = Field("ppo", title="Agent type")
    network_preset: NetworkPreset = Field("attention_memory", title="Network preset")
    gamma: float = Field(0.9999, ge=0.0, le=1.0, title="Gamma")
    vf_coef: float = Field(0.5, ge=0.0, title="Value coefficient")
    ent_coef: float = Field(0.05, ge=0.0, title="Entropy coefficient")
    eps_clip: float = Field(0.2, ge=0.0, le=1.0, title="PPO clip range")
    advantage_type: Literal["mc", "td0", "gae"] = Field("mc", title="Advantage estimator")
    gae_lambda: float = Field(0.95, ge=0.0, le=1.0, title="GAE lambda")
    normalize_advantages: bool = True
    aux_coef: float = Field(0.05, ge=0.0, title="Auxiliary loss coefficient")
    context_length: int = Field(64, ge=1, title="Spatiotemporal context length")
    device: str = Field("cpu", title="Torch device")
    network_config: dict[str, Any] | None = Field(
        None,
        title="Network overrides",
        description="Optional override for the network config dict; defaults to the preset.",
    )


class AgentRecord(BaseModel):
    """A persisted agent: config + checkpoint pointer."""

    id: str
    name: str
    config: AgentConfig
    checkpoint_path: str | None = None
    parent_run_id: str | None = None
    created_at: datetime
