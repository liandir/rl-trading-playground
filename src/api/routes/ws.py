"""WebSocket fan-out for live run and deployment events."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.services.events import get_broker, replay_history
from src.api.services.store import get_store
from src.api.settings import get_settings


router = APIRouter(tags=["ws"])


@router.websocket("/ws/runs/{run_id}")
async def run_events(ws: WebSocket, run_id: str) -> None:
    """Stream events for a training/validation ``run_id``."""

    if get_store().get_run(run_id) is None:
        await ws.close(code=4404)
        return
    await _stream(ws, run_id, kind="run")


@router.websocket("/ws/deployments/{deployment_id}")
async def deployment_events(ws: WebSocket, deployment_id: str) -> None:
    """Stream events for a live ``deployment_id``."""

    if get_store().get_deployment(deployment_id) is None:
        await ws.close(code=4404)
        return
    await _stream(ws, deployment_id, kind="deployment")


async def _stream(ws: WebSocket, entity_id: str, *, kind: str) -> None:
    await ws.accept()
    settings = get_settings()
    base_dir = settings.runs_dir if kind == "run" else settings.deployments_dir
    broker = get_broker()
    queue = await broker.subscribe(entity_id, kind=kind)  # type: ignore[arg-type]
    try:
        async for event in replay_history(entity_id, settings, base_dir=base_dir):
            await ws.send_json(event)
        while True:
            event = await queue.get()
            if event is None:
                break
            await ws.send_json(event)
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        raise
    finally:
        await broker.unsubscribe(entity_id, queue, kind=kind)  # type: ignore[arg-type]
        try:
            await ws.close()
        except Exception:
            pass
