"""WebSocket fan-out for live run and deployment events."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from rl_trading_playground.api.services.events import get_broker
from rl_trading_playground.api.services.store import get_store


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
    broker = get_broker()
    queue = await broker.subscribe(entity_id, kind=kind)  # type: ignore[arg-type]
    try:
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
