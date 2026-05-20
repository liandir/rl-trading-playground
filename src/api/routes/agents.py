"""CRUD endpoints for persisted agent presets."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.schemas.agent import AgentConfig, AgentRecord
from src.api.services.store import get_store


router = APIRouter(prefix="/agents", tags=["agents"])


class CreateAgentRequest(BaseModel):
    name: str = Field(..., min_length=1)
    config: AgentConfig


class AttachCheckpointRequest(BaseModel):
    checkpoint_path: str
    parent_run_id: str | None = None


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
