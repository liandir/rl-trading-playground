"""Checkpoint listing, tagging, and reading sidecar metadata."""
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
    tag: str | None


@router.get("", response_model=list[CheckpointRecord])
def list_checkpoints(
    run_id: str | None = Query(None),
    agent_id: str | None = Query(None),
) -> list[CheckpointRecord]:
    return get_store().list_checkpoints(run_id=run_id, agent_id=agent_id)


@router.get("/{checkpoint_id}/meta")
def get_meta(checkpoint_id: str) -> dict[str, Any]:
    rec = _find(checkpoint_id)
    meta_path = Path(rec.path).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"invalid meta json: {exc}") from exc


def _find(checkpoint_id: str) -> CheckpointRecord:
    matches = [cp for cp in get_store().list_checkpoints() if cp.id == checkpoint_id]
    if not matches:
        raise HTTPException(status_code=404, detail="checkpoint not found")
    return matches[0]
