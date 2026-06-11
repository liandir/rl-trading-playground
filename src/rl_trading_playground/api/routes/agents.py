"""CRUD endpoints for persisted agent presets, including .ptm import."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rl_trading_playground.api.schemas.agent import AgentConfig, AgentRecord
from rl_trading_playground.api.schemas.checkpoint import CheckpointMeta
from rl_trading_playground.api.services.store import get_store


router = APIRouter(prefix="/agents", tags=["agents"])


class CreateAgentRequest(BaseModel):
    name: str = Field(..., min_length=1)
    config: AgentConfig


class AttachCheckpointRequest(BaseModel):
    checkpoint_path: str
    parent_run_id: str | None = None


class InspectRequest(BaseModel):
    path: str = Field(..., description="Absolute path to a .ptm checkpoint file.")


class InspectResponse(BaseModel):
    has_sidecar: bool
    source_path: str
    sidecar_path: str | None = None
    suggested: AgentConfig | None = None
    parent_run_id: str | None = None
    saved_at: datetime | None = None
    network_keys: list[str] = Field(default_factory=list)
    note: str = ""


class ImportRequest(BaseModel):
    name: str = Field(..., min_length=1)
    source_path: str
    config: AgentConfig


@router.get("", response_model=list[AgentRecord])
def list_agents() -> list[AgentRecord]:
    return get_store().list_agents()


@router.post("", response_model=AgentRecord, status_code=201)
def create_agent(body: CreateAgentRequest) -> AgentRecord:
    return get_store().create_agent(body.name, body.config)


@router.get("/{agent_id}", response_model=AgentRecord)
def get_agent(agent_id: str) -> AgentRecord:
    rec = get_store().get_agent(agent_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return rec


@router.post("/{agent_id}/checkpoint", response_model=AgentRecord)
def attach_checkpoint(agent_id: str, body: AttachCheckpointRequest) -> AgentRecord:
    store = get_store()
    if store.get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail="agent not found")
    if not Path(body.checkpoint_path).exists():
        raise HTTPException(status_code=400, detail="checkpoint path does not exist")
    store.update_agent_checkpoint(agent_id, body.checkpoint_path, parent_run_id=body.parent_run_id)
    rec = store.get_agent(agent_id)
    assert rec is not None
    return rec


@router.delete("/{agent_id}", status_code=204)
def delete_agent(agent_id: str) -> None:
    if not get_store().delete_agent(agent_id):
        raise HTTPException(status_code=404, detail="agent not found")


@router.post("/inspect", response_model=InspectResponse)
def inspect_checkpoint(body: InspectRequest) -> InspectResponse:
    """Read a ``.ptm`` plus any sibling ``.meta.json`` sidecar.

    Returns suggested defaults the import form pre-fills. When no sidecar
    exists, the response includes the state_dict top-level keys so the user
    can manually pick the matching agent/network preset.
    """

    source = Path(body.path).expanduser()
    if not source.exists():
        raise HTTPException(status_code=404, detail="checkpoint file not found")
    if source.suffix not in {".ptm", ".pt", ".pth"}:
        raise HTTPException(status_code=400, detail="expected a .ptm/.pt/.pth file")

    sidecar = source.with_suffix(".meta.json")
    suggested: AgentConfig | None = None
    parent_run_id: str | None = None
    saved_at: datetime | None = None
    note = ""

    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text())
            meta = CheckpointMeta.model_validate(data)
            suggested = meta.agent_config
            parent_run_id = meta.parent_run_id
            saved_at = meta.saved_at
        except Exception as exc:
            note = f"sidecar present but unreadable: {exc}"

    network_keys: list[str] = []
    if suggested is None:
        # Try to peek at the state_dict so the import form can show *something*.
        try:
            import torch

            payload = torch.load(str(source), map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "network" in payload and hasattr(payload["network"], "keys"):
                network_keys = sorted(list(payload["network"].keys()))[:32]
            elif isinstance(payload, dict):
                network_keys = sorted(payload.keys())[:32]
            note = note or "no sidecar found; pick the matching agent and network preset by hand"
        except Exception as exc:
            note = note or f"failed to read torch payload: {exc}"

    return InspectResponse(
        has_sidecar=sidecar.exists() and suggested is not None,
        source_path=str(source),
        sidecar_path=str(sidecar) if sidecar.exists() else None,
        suggested=suggested,
        parent_run_id=parent_run_id,
        saved_at=saved_at,
        network_keys=network_keys,
        note=note,
    )


@router.post("/import", response_model=AgentRecord, status_code=201)
def import_agent(body: ImportRequest) -> AgentRecord:
    """Copy a ``.ptm`` into the agent's store dir and register it.

    The new agent's checkpoint sidecar embeds the (possibly user-edited)
    AgentConfig used to register it, plus a copy of the source sidecar
    if one existed, so the original hyperparameters remain auditable.
    """

    source = Path(body.source_path).expanduser()
    if not source.exists():
        raise HTTPException(status_code=404, detail="source path does not exist")
    if source.suffix not in {".ptm", ".pt", ".pth"}:
        raise HTTPException(status_code=400, detail="expected a .ptm/.pt/.pth file")

    store = get_store()
    rec = store.create_agent(body.name, body.config)
    agent_dir = store.settings.agents_dir / rec.id
    agent_dir.mkdir(parents=True, exist_ok=True)
    target = agent_dir / "checkpoint.ptm"
    shutil.copy2(source, target)

    original_meta: dict[str, Any] | None = None
    source_sidecar = source.with_suffix(".meta.json")
    if source_sidecar.exists():
        try:
            original_meta = json.loads(source_sidecar.read_text())
        except Exception:
            original_meta = None

    original: CheckpointMeta | None = None
    if isinstance(original_meta, dict):
        try:
            original = CheckpointMeta.model_validate(original_meta)
        except Exception:
            original = None

    meta = CheckpointMeta(
        agent_config=body.config,
        # Carry over the training data/env snapshots so compatibility checks
        # (e.g. bar-interval guards on deployments) keep working post-import.
        data_config=original.data_config if original else None,
        env_config=original.env_config if original else None,
        step=original.step if original else 0,
        episode=original.episode if original else 0,
        metric_name=original.metric_name if original else None,
        metric_value=original.metric_value if original else None,
        parent_run_id=None,
        saved_at=datetime.now(timezone.utc),
        extra={
            "source_path": str(source),
            "imported_from": original_meta,
        },
    )
    (agent_dir / "checkpoint.meta.json").write_text(meta.model_dump_json(indent=2))

    store.update_agent_checkpoint(rec.id, str(target), parent_run_id=None)
    store.record_checkpoint(
        run_id=None,
        agent_id=rec.id,
        step=meta.step,
        path=str(target),
        metric_name=meta.metric_name,
        metric_value=meta.metric_value,
        tag="imported",
    )

    refreshed = store.get_agent(rec.id)
    assert refreshed is not None
    return refreshed
