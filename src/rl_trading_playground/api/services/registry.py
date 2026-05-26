"""Static registry of choices the UI uses to populate forms.

This is a thin, side-effect-free module so the FastAPI process can answer
`GET /registry` without importing torch or any heavy domain code. The runner
subprocess is where actual agent/network/env classes get constructed.
"""
from __future__ import annotations

from typing import Any

from rl_trading_playground.api.schemas.registry import (
    AgentTypeInfo,
    EnvironmentTypeInfo,
    NetworkPresetInfo,
    RegistryResponse,
)


AGENT_TYPES: tuple[AgentTypeInfo, ...] = (
    AgentTypeInfo(name="ppo", family="ppo", description="Hierarchical PPO."),
    AgentTypeInfo(name="aac", family="aac", description="Hierarchical advantage actor-critic."),
    AgentTypeInfo(name="aaq", family="aaq", description="Hierarchical advantage actor + Q-critic."),
    AgentTypeInfo(name="ppo_aux", family="ppo", supports_aux=True, description="PPO with auxiliary heads."),
    AgentTypeInfo(name="aac_aux", family="aac", supports_aux=True, description="AAC with auxiliary heads."),
    AgentTypeInfo(name="aaq_aux", family="aaq", supports_aux=True, description="AAQ with auxiliary heads."),
    AgentTypeInfo(name="aac_model", family="aac", description="Model-based AAC with EMA encoder."),
    AgentTypeInfo(
        name="spatiotemporal_aac",
        family="aac",
        supports_aux=True,
        description="Spatiotemporal attention AAC.",
    ),
)


NETWORK_PRESETS: tuple[NetworkPresetInfo, ...] = (
    NetworkPresetInfo(
        name="tiny",
        network_type="flat_per_asset_action_value",
        description="Smallest preset; useful for tests and smoke runs.",
    ),
    NetworkPresetInfo(name="flat_per_asset", network_type="flat_per_asset_action_value"),
    NetworkPresetInfo(name="attention_memory", network_type="attention_memory_action_value"),
    NetworkPresetInfo(name="auxiliary_per_asset", network_type="auxiliary_per_asset_action_value"),
    NetworkPresetInfo(name="flat_model", network_type="flat_per_asset_model"),
    NetworkPresetInfo(name="attention_model", network_type="attention_memory_model"),
    NetworkPresetInfo(
        name="spatiotemporal_auxiliary",
        network_type="spatiotemporal_aux_per_asset_action_value",
    ),
)


ENVIRONMENT_TYPES: tuple[EnvironmentTypeInfo, ...] = (
    EnvironmentTypeInfo(
        name="longshort_hierarchical_leverage",
        description="Hierarchical long/short env with leverage and bucketed sizing.",
    ),
)


DATA_SOURCES: tuple[str, ...] = ("auto", "prepared", "kraken_csv", "synthetic")


_NETWORK_SCHEMAS: dict[str, dict[str, Any]] = {
    "tiny": {
        "hidden_dims_asset": [16],
        "d_model": 8,
        "hidden_dims_mem": [16],
        "hidden_dims_ff": [16],
        "d_mem": 8,
        "d_ff": 8,
    },
}


def network_schema(preset: str) -> dict[str, Any]:
    """Return the JSON-friendly shape of overridable fields for a preset."""

    return _NETWORK_SCHEMAS.get(preset, {})


def response() -> RegistryResponse:
    """Build the payload served at `GET /registry`."""

    return RegistryResponse(
        agent_types=list(AGENT_TYPES),
        network_presets=list(NETWORK_PRESETS),
        environment_types=list(ENVIRONMENT_TYPES),
        data_sources=list(DATA_SOURCES),
        network_schemas={p.name: network_schema(p.name) for p in NETWORK_PRESETS},
    )
