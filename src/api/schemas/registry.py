"""Registry response: choices the UI uses to populate dropdowns."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class NetworkPresetInfo(BaseModel):
    name: str
    network_type: str
    description: str = ""


class AgentTypeInfo(BaseModel):
    name: str
    family: str
    supports_aux: bool = False
    description: str = ""


class EnvironmentTypeInfo(BaseModel):
    name: str
    description: str = ""


class RegistryResponse(BaseModel):
    """Everything the frontend needs to render selection UI."""

    agent_types: list[AgentTypeInfo]
    network_presets: list[NetworkPresetInfo]
    environment_types: list[EnvironmentTypeInfo]
    data_sources: list[str]
    network_schemas: dict[str, dict[str, Any]] = {}
