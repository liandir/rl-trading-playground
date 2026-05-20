"""Subprocess lifecycle: spawn, track, signal, harvest exit codes.

A single :class:`RunnerService` instance is shared across the FastAPI app.
It owns the table of live runner subprocesses keyed by ``run_id`` and a
single background task that reaps exited processes so their exit codes are
recorded on the run row.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path

from src.api.services.store import Store, get_store
from src.api.settings import Settings, get_settings


class RunnerService:
    """Spawns runner subprocesses and persists their lifecycle in the store."""

    def __init__(self, settings: Settings | None = None, store: Store | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store()
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._stdout_files: dict[str, object] = {}
        self._lock = asyncio.Lock()
        self._reaper_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.create_task(self._reaper_loop(), name="runner-reaper")

    async def stop(self) -> None:
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None
        for run_id in list(self._procs):
            await self.signal_stop(run_id)
        for fh in self._stdout_files.values():
            try:
                fh.close()  # type: ignore[union-attr]
            except Exception:
                pass
        self._procs.clear()
        self._stdout_files.clear()

    async def launch(self, run_id: str) -> int:
        """Spawn the runner subprocess for ``run_id`` and record its PID."""

        run_dir = self.settings.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.log"

        env = os.environ.copy()
        env["STUDIO_STORE"] = str(self.settings.store_root)
        repo_root = Path(__file__).resolve().parents[3]
        fh = stdout_path.open("ab", buffering=0)
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.api.runner", run_id],
            cwd=repo_root,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        async with self._lock:
            self._procs[run_id] = proc
            self._stdout_files[run_id] = fh
        self.store.update_run_status(run_id, "running", pid=proc.pid)
        return proc.pid

    async def signal_stop(self, run_id: str) -> bool:
        """Send SIGTERM to a tracked subprocess. Returns False if unknown."""

        async with self._lock:
            proc = self._procs.get(run_id)
        if proc is None or proc.poll() is not None:
            return False
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True

    async def _reaper_loop(self) -> None:
        """Poll tracked subprocesses, persist exit codes and clean up handles."""

        try:
            while True:
                await asyncio.sleep(0.25)
                done: list[tuple[str, int]] = []
                async with self._lock:
                    for run_id, proc in list(self._procs.items()):
                        rc = proc.poll()
                        if rc is not None:
                            done.append((run_id, rc))
                            self._procs.pop(run_id, None)
                            fh = self._stdout_files.pop(run_id, None)
                            if fh is not None:
                                try:
                                    fh.close()  # type: ignore[union-attr]
                                except Exception:
                                    pass
                for run_id, rc in done:
                    self._on_exit(run_id, rc)
        except asyncio.CancelledError:
            raise

    def _on_exit(self, run_id: str, rc: int) -> None:
        """Reconcile run status with the actual subprocess exit code."""

        rec = self.store.get_run(run_id)
        if rec is None:
            return
        if rec.status in ("complete", "stopped", "failed"):
            self.store.update_run_status(run_id, rec.status, exit_code=rc)
            return
        # Killed by SIGTERM / SIGINT before the runner could mark itself stopped.
        if rc in (-signal.SIGTERM, -signal.SIGINT):
            self.store.update_run_status(run_id, "stopped", exit_code=rc)
            return
        status = "failed" if rc != 0 else "complete"
        notes = f"exit code {rc}" if rc != 0 else None
        self.store.update_run_status(run_id, status, exit_code=rc, notes=notes)


_runner: RunnerService | None = None


def get_runner() -> RunnerService:
    global _runner
    if _runner is None:
        _runner = RunnerService()
    return _runner


def reset_runner_for_tests() -> None:
    global _runner
    _runner = None
