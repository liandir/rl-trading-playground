"""Deployment lifecycle: create (spawns live runner), list, stop, replay events."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from src.api.schemas.deployment import CreateDeploymentRequest, DeploymentRecord
from src.api.services.compat import check_interval
from src.api.services.events import replay_history
from src.api.services.runner import get_runner
from src.api.services.store import get_store
from src.api.settings import get_settings


router = APIRouter(prefix="/deployments", tags=["deployments"])


@router.get("", response_model=list[DeploymentRecord])
def list_deployments(
    status: str | None = Query(None, pattern="^(queued|running|complete|stopped|failed)$"),
) -> list[DeploymentRecord]:
    return get_store().list_deployments(status=status)


@router.post("", response_model=DeploymentRecord, status_code=201)
async def create_deployment(body: CreateDeploymentRequest) -> DeploymentRecord:
    store = get_store()
    agent = store.get_agent(body.spec.agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent '{body.spec.agent_id}' not found")
    if not agent.checkpoint_path:
        raise HTTPException(status_code=400, detail="agent has no checkpoint attached")

    check = check_interval(agent, body.spec.interval_minutes)
    if not check.ok and not body.spec.override_interval_mismatch:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "interval_mismatch",
                "message": check.message,
                "training_interval": check.training_interval,
                "new_interval": check.new_interval,
            },
        )

    rec = store.create_deployment(body.spec, name=body.name)
    runner = get_runner()
    await runner.start()
    await runner.launch_deployment(rec.id)
    refreshed = store.get_deployment(rec.id)
    assert refreshed is not None
    return refreshed


@router.get("/{deployment_id}", response_model=DeploymentRecord)
def get_deployment(deployment_id: str) -> DeploymentRecord:
    rec = get_store().get_deployment(deployment_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="deployment not found")
    return rec


@router.post("/{deployment_id}/stop", response_model=DeploymentRecord)
async def stop_deployment(deployment_id: str) -> DeploymentRecord:
    store = get_store()
    rec = store.get_deployment(deployment_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="deployment not found")
    runner = get_runner()
    if not await runner.signal_stop(deployment_id):
        raise HTTPException(status_code=409, detail="deployment is not active")
    refreshed = store.get_deployment(deployment_id)
    assert refreshed is not None
    return refreshed


@router.get("/{deployment_id}/events")
async def list_events(deployment_id: str) -> JSONResponse:
    store = get_store()
    if store.get_deployment(deployment_id) is None:
        raise HTTPException(status_code=404, detail="deployment not found")
    settings = get_settings()
    events: list[dict[str, Any]] = []
    async for event in replay_history(
        deployment_id, settings, base_dir=settings.deployments_dir
    ):
        events.append(event)
    return JSONResponse(events)
