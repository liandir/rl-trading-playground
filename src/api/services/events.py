"""Tail run events from disk and fan them out to WebSocket subscribers.

Each run has at most one tail task. Tail tasks are started on demand the
first time a subscriber attaches and stay alive until the run reaches a
terminal status *and* no new lines have been observed for one extra poll.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

from src.api.services.store import Store, get_store
from src.api.settings import Settings, get_settings


_TERMINAL = {"complete", "stopped", "failed"}


class EventBroker:
    """Coordinates one tail task per run plus a set of subscriber queues."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: Store | None = None,
        *,
        poll_interval: float = 0.1,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store()
        self.poll_interval = poll_interval
        self._tail_tasks: dict[str, asyncio.Task[None]] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any] | None]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, run_id: str) -> asyncio.Queue[dict[str, Any] | None]:
        """Attach a subscriber queue and ensure a tail task is running."""

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=512)
        async with self._lock:
            self._subscribers.setdefault(run_id, set()).add(queue)
            if run_id not in self._tail_tasks or self._tail_tasks[run_id].done():
                self._tail_tasks[run_id] = asyncio.create_task(
                    self._tail_loop(run_id), name=f"event-tail-{run_id}"
                )
        return queue

    async def unsubscribe(self, run_id: str, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        async with self._lock:
            subs = self._subscribers.get(run_id)
            if subs is not None:
                subs.discard(queue)
                if not subs:
                    self._subscribers.pop(run_id, None)

    async def shutdown(self) -> None:
        async with self._lock:
            tasks = list(self._tail_tasks.values())
            self._tail_tasks.clear()
            self._subscribers.clear()
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _tail_loop(self, run_id: str) -> None:
        """Read appended JSONL lines and push each parsed event to subscribers."""

        path = self.settings.runs_dir / run_id / "events.jsonl"
        # Wait for the file to appear, but give up after a few seconds.
        for _ in range(50):
            if path.exists():
                break
            await asyncio.sleep(self.poll_interval)
        offset = 0
        try:
            while True:
                offset, lines = await asyncio.to_thread(_read_new_lines, path, offset)
                for line in lines:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    await self._broadcast(run_id, event)
                # If the run reached a terminal status and no new lines arrived,
                # do one more read after a short wait and then exit.
                status = self._run_status(run_id)
                if status in _TERMINAL and not lines:
                    await asyncio.sleep(self.poll_interval)
                    offset, trailing = await asyncio.to_thread(_read_new_lines, path, offset)
                    for line in trailing:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        await self._broadcast(run_id, event)
                    await self._broadcast(run_id, None)
                    return
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            await self._broadcast(run_id, None)
            raise

    async def _broadcast(self, run_id: str, event: dict[str, Any] | None) -> None:
        async with self._lock:
            subscribers = list(self._subscribers.get(run_id, ()))
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest item and retry.
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    continue

    def _run_status(self, run_id: str) -> str | None:
        rec = self.store.get_run(run_id)
        return rec.status if rec is not None else None


def _read_new_lines(path: Path, offset: int) -> tuple[int, list[str]]:
    """Read all complete lines after ``offset``; returns ``(new_offset, lines)``."""

    if not path.exists():
        return offset, []
    with path.open("rb") as fh:
        fh.seek(offset)
        buf = fh.read()
    if not buf:
        return offset, []
    last_nl = buf.rfind(b"\n")
    if last_nl < 0:
        return offset, []
    consumed = buf[: last_nl + 1]
    text = consumed.decode("utf-8", errors="replace")
    lines = [line for line in text.split("\n") if line.strip()]
    return offset + len(consumed), lines


async def replay_history(run_id: str, settings: Settings) -> AsyncIterator[dict[str, Any]]:
    """Yield every event currently on disk for ``run_id``."""

    path = settings.runs_dir / run_id / "events.jsonl"
    if not path.exists():
        return
    text = await asyncio.to_thread(path.read_text)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


_broker: EventBroker | None = None


def get_broker() -> EventBroker:
    global _broker
    if _broker is None:
        _broker = EventBroker()
    return _broker


def reset_broker_for_tests() -> None:
    global _broker
    _broker = None
