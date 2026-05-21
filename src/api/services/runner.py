"""Subprocess lifecycle: spawn, track, signal, harvest exit codes.

Handles both training/validation runs and live paper-trading deployments.
A single reaper task wakes up periodically to reconcile exit codes with
the matching row in the store.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.api.services.store import Store, get_store
from src.api.settings import Settings, get_settings


ProcKind = Literal["run", "deployment"]


@dataclass
class _Proc:
    kind: ProcKind
    proc: "subprocess.Popen[bytes]"
    stdout: object


class RunnerService:
    """Spawns runner subprocesses and persists their lifecycle in the store."""

    def __init__(self, settings: Settings | None = None, store: Store | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store()
        self._procs: dict[str, _Proc] = {}
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
        for proc_id in list(self._procs):
            await self.signal_stop(proc_id)
        for entry in self._procs.values():
            try:
                entry.stdout.close()  # type: ignore[union-attr]
            except Exception:
                pass
        self._procs.clear()

    async def launch(self, run_id: str) -> int:
        """Spawn the training/validation runner subprocess for ``run_id``."""

        pid = await self._spawn(
            kind="run",
            entity_id=run_id,
            module="src.api.runner",
            log_dir=self.settings.runs_dir / run_id,
        )
        self.store.update_run_status(run_id, "running", pid=pid)
        return pid

    async def launch_deployment(self, deployment_id: str) -> int:
        """Spawn the live deployment subprocess for ``deployment_id``."""

        pid = await self._spawn(
            kind="deployment",
            entity_id=deployment_id,
            module="src.api.live",
            log_dir=self.settings.deployments_dir / deployment_id,
        )
        self.store.update_deployment_status(deployment_id, "running", pid=pid)
        return pid

    async def signal_stop(self, entity_id: str) -> bool:
        """Send SIGTERM to a tracked subprocess. Returns False if unknown."""

        async with self._lock:
            entry = self._procs.get(entity_id)
        if entry is None or entry.proc.poll() is not None:
            return False
        try:
            entry.proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True

    async def _spawn(
        self, *, kind: ProcKind, entity_id: str, module: str, log_dir: Path
    ) -> int:
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        env = os.environ.copy()
        env["STUDIO_STORE"] = str(self.settings.store_root)
        repo_root = Path(__file__).resolve().parents[3]
        fh = stdout_path.open("ab", buffering=0)
        proc = subprocess.Popen(
            [sys.executable, "-m", module, entity_id],
            cwd=repo_root,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        async with self._lock:
            self._procs[entity_id] = _Proc(kind=kind, proc=proc, stdout=fh)
        return proc.pid

    async def _reaper_loop(self) -> None:
        """Poll tracked subprocesses, persist exit codes and clean up handles."""

        try:
            while True:
                await asyncio.sleep(0.25)
                done: list[tuple[str, ProcKind, int]] = []
                async with self._lock:
                    for entity_id, entry in list(self._procs.items()):
                        rc = entry.proc.poll()
                        if rc is not None:
                            done.append((entity_id, entry.kind, rc))
                            self._procs.pop(entity_id, None)
                            try:
                                entry.stdout.close()  # type: ignore[union-attr]
                            except Exception:
                                pass
                for entity_id, kind, rc in done:
                    self._on_exit(entity_id, kind, rc)
        except asyncio.CancelledError:
            raise

    def _on_exit(self, entity_id: str, kind: ProcKind, rc: int) -> None:
        if kind == "run":
            rec = self.store.get_run(entity_id)
            if rec is None:
                return
            if rec.status in ("complete", "stopped", "failed"):
                self.store.update_run_status(entity_id, rec.status, exit_code=rc)
                return
            if rc in (-signal.SIGTERM, -signal.SIGINT):
                self.store.update_run_status(entity_id, "stopped", exit_code=rc)
                return
            status = "failed" if rc != 0 else "complete"
            notes = f"exit code {rc}" if rc != 0 else None
            self.store.update_run_status(entity_id, status, exit_code=rc, notes=notes)
        else:  # deployment
            rec = self.store.get_deployment(entity_id)
            if rec is None:
                return
            if rec.status in ("complete", "stopped", "failed"):
                self.store.update_deployment_status(entity_id, rec.status, exit_code=rc)
                return
            if rc in (-signal.SIGTERM, -signal.SIGINT):
                self.store.update_deployment_status(entity_id, "stopped", exit_code=rc)
                return
            status = "failed" if rc != 0 else "complete"
            notes = f"exit code {rc}" if rc != 0 else None
            self.store.update_deployment_status(entity_id, status, exit_code=rc, notes=notes)


_runner: RunnerService | None = None


def get_runner() -> RunnerService:
    global _runner
    if _runner is None:
        _runner = RunnerService()
    return _runner


def reset_runner_for_tests() -> None:
    global _runner
    _runner = None
