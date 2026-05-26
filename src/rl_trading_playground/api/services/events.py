"""Tail JSONL events from disk and fan them out to WebSocket subscribers.

Handles both runs and deployments: each subscription is keyed by an opaque
entity id and the directory that holds its ``events.jsonl``. One tail task
per (kind, id) pair, kept alive until the entity reaches a terminal status
and no new lines have been observed for one extra poll.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from rl_trading_playground.api.services.store import Store, get_store
from rl_trading_playground.api.settings import Settings, get_settings


_TERMINAL = {"complete", "stopped", "failed"}

EntityKind = Literal["run", "deployment"]


class EventBroker:
    """Coordinates per-entity tail tasks and subscriber queues."""

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
        self._tail_tasks: dict[tuple[EntityKind, str], asyncio.Task[None]] = {}
        self._tail_offsets: dict[tuple[EntityKind, str], int] = {}
        self._subscribers: dict[
            tuple[EntityKind, str], set[asyncio.Queue[dict[str, Any] | None]]
        ] = {}
        self._lock = asyncio.Lock()

    async def subscribe(
        self, entity_id: str, *, kind: EntityKind = "run"
    ) -> asyncio.Queue[dict[str, Any] | None]:
        """Attach a subscriber queue, replay history into it, and start tailing live events."""

        key: tuple[EntityKind, str] = (kind, entity_id)
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=8192)
        path = _events_path(self.settings, entity_id, kind=kind)
        async with self._lock:
            existing = self._tail_tasks.get(key)
            if existing is None or existing.done():
                snapshot_end = path.stat().st_size if path.exists() else 0
                self._tail_offsets[key] = snapshot_end
                self._tail_tasks[key] = asyncio.create_task(
                    self._tail_loop(key), name=f"event-tail-{kind}-{entity_id}"
                )
            else:
                snapshot_end = self._tail_offsets.get(key, 0)
            self._subscribers.setdefault(key, set()).add(queue)
        # Outside the lock: replay historical events (up to the tail task's
        # starting offset) into this subscriber's queue. The tail task then
        # forwards only events past that offset, so the subscriber sees the
        # full sequence exactly once regardless of how many other subscribers
        # exist.
        if snapshot_end > 0 and path.exists():
            try:
                data = await asyncio.to_thread(_read_bytes, path, snapshot_end)
            except OSError:
                data = b""
            for raw in data.split(b"\n"):
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    break
        return queue

    async def unsubscribe(
        self,
        entity_id: str,
        queue: asyncio.Queue[dict[str, Any] | None],
        *,
        kind: EntityKind = "run",
    ) -> None:
        key = (kind, entity_id)
        task_to_cancel: asyncio.Task[None] | None = None
        async with self._lock:
            subs = self._subscribers.get(key)
            if subs is not None:
                subs.discard(queue)
                if not subs:
                    self._subscribers.pop(key, None)
                    self._tail_offsets.pop(key, None)
                    task = self._tail_tasks.pop(key, None)
                    if task is not None and not task.done():
                        task_to_cancel = task
        if task_to_cancel is not None:
            task_to_cancel.cancel()

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

    async def _tail_loop(self, key: tuple[EntityKind, str]) -> None:
        kind, entity_id = key
        path = _events_path(self.settings, entity_id, kind=kind)
        for _ in range(50):
            if path.exists():
                break
            await asyncio.sleep(self.poll_interval)
        try:
            while True:
                offset = self._tail_offsets.get(key, 0)
                new_offset, lines = await asyncio.to_thread(_read_new_lines, path, offset)
                self._tail_offsets[key] = new_offset
                for line in lines:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    await self._broadcast(key, event)
                status = self._status(kind, entity_id)
                if status in _TERMINAL and not lines:
                    await asyncio.sleep(self.poll_interval)
                    offset = self._tail_offsets.get(key, 0)
                    final_offset, trailing = await asyncio.to_thread(_read_new_lines, path, offset)
                    self._tail_offsets[key] = final_offset
                    for line in trailing:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        await self._broadcast(key, event)
                    await self._broadcast(key, None)
                    return
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            raise

    async def _broadcast(
        self, key: tuple[EntityKind, str], event: dict[str, Any] | None
    ) -> None:
        async with self._lock:
            subscribers = list(self._subscribers.get(key, ()))
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    continue

    def _status(self, kind: EntityKind, entity_id: str) -> str | None:
        if kind == "run":
            rec = self.store.get_run(entity_id)
        else:
            rec = self.store.get_deployment(entity_id)
        return rec.status if rec is not None else None


def _events_path(settings: Settings, entity_id: str, *, kind: EntityKind) -> Path:
    base = settings.runs_dir if kind == "run" else settings.deployments_dir
    return base / entity_id / "events.jsonl"


def _read_bytes(path: Path, end: int) -> bytes:
    """Read the first ``end`` bytes of ``path``."""
    with path.open("rb") as fh:
        return fh.read(end)


def _read_new_lines(path: Path, offset: int) -> tuple[int, list[str]]:
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


async def replay_history(
    entity_id: str,
    settings: Settings,
    *,
    base_dir: Path | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield every event currently on disk for ``entity_id``.

    By default reads from ``runs/<id>/events.jsonl``; pass ``base_dir`` to
    target deployments or any other on-disk JSONL stream.
    """

    base = base_dir or settings.runs_dir
    path = base / entity_id / "events.jsonl"
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
