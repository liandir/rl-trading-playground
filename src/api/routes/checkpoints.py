"""Checkpoint listing, tagging, attaching to agents, and sidecar metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.api.schemas.checkpoint import CheckpointRecord
from src.api.services.store import get_store


router = APIRouter(prefix="/checkpoints", tags=["checkpoints"])


class TagRequest(BaseModel):
    tag: str | None = None


class AttachRequest(BaseModel):
    agent_id: str


@router.get("", response_model=list[CheckpointRecord])
def list_checkpoints(
    run_id: str | None = Query(None),
    agent_id: str | None = Query(None),
) -> list[CheckpointRecord]:
    return get_store().list_checkpoints(run_id=run_id, agent_id=agent_id)


@router.get("/{checkpoint_id}", response_model=CheckpointRecord)
def get_checkpoint(checkpoint_id: str) -> CheckpointRecord:
    rec = get_store().get_checkpoint(checkpoint_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="checkpoint not found")
    return rec


@router.get("/{checkpoint_id}/meta")
def get_meta(checkpoint_id: str) -> dict[str, Any]:
    rec = _require_checkpoint(checkpoint_id)
    meta_path = Path(rec.path).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"invalid meta json: {exc}") from exc


@router.post("/{checkpoint_id}/tag", response_model=CheckpointRecord)
def set_tag(checkpoint_id: str, body: TagRequest) -> CheckpointRecord:
    store = get_store()
    if not store.set_checkpoint_tag(checkpoint_id, body.tag):
        raise HTTPException(status_code=404, detail="checkpoint not found")
    rec = store.get_checkpoint(checkpoint_id)
    assert rec is not None
    return rec


@router.post("/{checkpoint_id}/attach", response_model=CheckpointRecord)
def attach_to_agent(checkpoint_id: str, body: AttachRequest) -> CheckpointRecord:
    """Point an agent at this checkpoint's path so the next run loads it."""

    store = get_store()
    rec = _require_checkpoint(checkpoint_id)
    if not Path(rec.path).exists():
        raise HTTPException(status_code=400, detail="checkpoint file no longer exists on disk")
    if store.get_agent(body.agent_id) is None:
        raise HTTPException(status_code=404, detail="agent not found")
    store.update_agent_checkpoint(body.agent_id, rec.path, parent_run_id=rec.run_id)
    return rec


def _require_checkpoint(checkpoint_id: str) -> CheckpointRecord:
    rec = get_store().get_checkpoint(checkpoint_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="checkpoint not found")
    return rec
