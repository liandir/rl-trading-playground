"""Run lifecycle: create (which spawns the subprocess), list, stop, replay events."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from src.api.schemas.run import CreateRunRequest, RunRecord
from src.api.services.events import replay_history
from src.api.services.runner import get_runner
from src.api.services.store import get_store
from src.api.settings import get_settings


router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("", response_model=list[RunRecord])
def list_runs(
    kind: str | None = Query(None, pattern="^(training|validation)$"),
    status: str | None = Query(None, pattern="^(queued|running|complete|stopped|failed)$"),
) -> list[RunRecord]:
    return get_store().list_runs(kind=kind, status=status)


@router.post("", response_model=RunRecord, status_code=201)
async def create_run(body: CreateRunRequest) -> RunRecord:
    store = get_store()
    rec = store.create_run(body.spec, name=body.name)
    runner = get_runner()
    await runner.start()
    await runner.launch(rec.id)
    refreshed = store.get_run(rec.id)
    assert refreshed is not None
    return refreshed


@router.get("/{run_id}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    rec = get_store().get_run(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    return rec


@router.post("/{run_id}/stop", response_model=RunRecord)
async def stop_run(run_id: str) -> RunRecord:
    store = get_store()
    rec = store.get_run(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    runner = get_runner()
    if not await runner.signal_stop(run_id):
        raise HTTPException(status_code=409, detail="run is not active")
    refreshed = store.get_run(run_id)
    assert refreshed is not None
    return refreshed


@router.get("/{run_id}/events")
async def list_events(run_id: str) -> JSONResponse:
    """Return every event currently persisted for a run as a JSON array."""

    store = get_store()
    if store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    events: list[dict[str, Any]] = []
    async for event in replay_history(run_id, get_settings()):
        events.append(event)
    return JSONResponse(events)


@router.get("/{run_id}/checkpoints")
def run_checkpoints(run_id: str) -> list[dict[str, Any]]:
    if get_store().get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return [cp.model_dump() for cp in get_store().list_checkpoints(run_id=run_id)]


@router.get("/{run_id}/artifact")
def run_artifact(run_id: str, name: str = Query("validation")) -> JSONResponse:
    """Return a JSON artifact written by the runner under ``artifacts/<name>.json``."""

    store = get_store()
    if store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    artifact_path = store.run_dir(run_id) / "artifacts" / f"{name}.json"
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail=f"artifact '{name}' not found for run")
    try:
        payload = json.loads(artifact_path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"invalid artifact json: {exc}") from exc
    return JSONResponse(payload)
