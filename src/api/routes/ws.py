"""WebSocket fan-out for live run events."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.services.events import get_broker, replay_history
from src.api.services.store import get_store
from src.api.settings import get_settings


router = APIRouter(tags=["ws"])


@router.websocket("/ws/runs/{run_id}")
async def run_events(ws: WebSocket, run_id: str) -> None:
    """Stream every event for ``run_id`` as JSON messages over the WebSocket.

    The first batch replays everything currently on disk, then the broker
    forwards new lines as the runner subprocess appends them. A ``null``
    payload (sent once) signals "no more events; close the socket".
    """

    store = get_store()
    if store.get_run(run_id) is None:
        await ws.close(code=4404)
        return

    await ws.accept()
    settings = get_settings()
    broker = get_broker()
    queue = await broker.subscribe(run_id)
    try:
        async for event in replay_history(run_id, settings):
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
        await broker.unsubscribe(run_id, queue)
        try:
            await ws.close()
        except Exception:
            pass
