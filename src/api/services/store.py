"""SQLite + filesystem store for studio entities.

SQLite holds indexable metadata (agents, envs, runs, checkpoints). Heavy
artifacts (config snapshots, checkpoints, event streams, logs) live under
`<store>/<kind>/<id>/` on disk. All writes flow through this module so routes
never touch the filesystem directly.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from uuid import uuid4

from src.api.schemas.agent import AgentConfig, AgentRecord
from src.api.schemas.checkpoint import CheckpointRecord
from src.api.schemas.data import DataConfig, DataSourceRecord
from src.api.schemas.env import EnvironmentConfig, EnvironmentRecord
from src.api.schemas.run import RunRecord, RunSpec, RunStatus
from src.api.settings import Settings, get_settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    network_preset TEXT NOT NULL,
    checkpoint_path TEXT,
    parent_run_id TEXT,
    created_at TEXT NOT NULL,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS envs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    env_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    agent_id TEXT,
    env_id TEXT,
    parent_run_id TEXT,
    pid INTEGER,
    exit_code INTEGER,
    notes TEXT NOT NULL DEFAULT '',
    spec_json TEXT,
    summary_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS runs_status_idx ON runs(status);
CREATE INDEX IF NOT EXISTS runs_kind_idx ON runs(kind);

CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    agent_id TEXT,
    step INTEGER NOT NULL,
    metric_name TEXT,
    metric_value REAL,
    tag TEXT,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS checkpoints_run_idx ON checkpoints(run_id);
CREATE INDEX IF NOT EXISTS checkpoints_agent_idx ON checkpoints(agent_id);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def new_id() -> str:
    return uuid4().hex[:12]


class Store:
    """Process-local store. One instance is shared across the FastAPI app."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.settings.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ----- low level helpers -------------------------------------------------

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            finally:
                cur.close()

    # ----- agents ------------------------------------------------------------

    def create_agent(self, name: str, config: AgentConfig) -> AgentRecord:
        rec = AgentRecord(
            id=new_id(),
            name=name,
            config=config,
            checkpoint_path=None,
            parent_run_id=None,
            created_at=_now(),
        )
        agent_dir = self.settings.agents_dir / rec.id
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "config.json").write_text(rec.model_dump_json(indent=2))
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO agents (id, name, agent_type, network_preset, checkpoint_path,"
                " parent_run_id, created_at, config_json) VALUES (?,?,?,?,?,?,?,?)",
                (
                    rec.id,
                    rec.name,
                    config.agent_type,
                    config.network_preset,
                    None,
                    None,
                    _iso(rec.created_at),
                    config.model_dump_json(),
                ),
            )
        return rec

    def list_agents(self) -> list[AgentRecord]:
        with self._cursor() as cur:
            rows = cur.execute("SELECT * FROM agents ORDER BY created_at DESC").fetchall()
        return [self._row_to_agent(row) for row in rows]

    def get_agent(self, agent_id: str) -> AgentRecord | None:
        with self._cursor() as cur:
            row = cur.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return self._row_to_agent(row) if row else None

    def update_agent_checkpoint(self, agent_id: str, path: str, parent_run_id: str | None = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE agents SET checkpoint_path = ?, parent_run_id = COALESCE(?, parent_run_id)"
                " WHERE id = ?",
                (path, parent_run_id, agent_id),
            )

    def delete_agent(self, agent_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
            deleted = cur.rowcount > 0
        agent_dir = self.settings.agents_dir / agent_id
        if agent_dir.exists():
            _rmtree(agent_dir)
        return deleted

    def _row_to_agent(self, row: sqlite3.Row) -> AgentRecord:
        return AgentRecord(
            id=row["id"],
            name=row["name"],
            config=AgentConfig.model_validate_json(row["config_json"]),
            checkpoint_path=row["checkpoint_path"],
            parent_run_id=row["parent_run_id"],
            created_at=_parse_iso(row["created_at"]) or _now(),
        )

    # ----- environments ------------------------------------------------------

    def create_env(self, name: str, config: EnvironmentConfig) -> EnvironmentRecord:
        rec = EnvironmentRecord(id=new_id(), name=name, config=config, created_at=_now())
        env_dir = self.settings.envs_dir / rec.id
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "config.json").write_text(rec.model_dump_json(indent=2))
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO envs (id, name, env_type, created_at, config_json) VALUES (?,?,?,?,?)",
                (rec.id, rec.name, config.env_type, _iso(rec.created_at), config.model_dump_json()),
            )
        return rec

    def list_envs(self) -> list[EnvironmentRecord]:
        with self._cursor() as cur:
            rows = cur.execute("SELECT * FROM envs ORDER BY created_at DESC").fetchall()
        return [self._row_to_env(row) for row in rows]

    def get_env(self, env_id: str) -> EnvironmentRecord | None:
        with self._cursor() as cur:
            row = cur.execute("SELECT * FROM envs WHERE id = ?", (env_id,)).fetchone()
        return self._row_to_env(row) if row else None

    def delete_env(self, env_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute("DELETE FROM envs WHERE id = ?", (env_id,))
            deleted = cur.rowcount > 0
        env_dir = self.settings.envs_dir / env_id
        if env_dir.exists():
            _rmtree(env_dir)
        return deleted

    def _row_to_env(self, row: sqlite3.Row) -> EnvironmentRecord:
        return EnvironmentRecord(
            id=row["id"],
            name=row["name"],
            config=EnvironmentConfig.model_validate_json(row["config_json"]),
            created_at=_parse_iso(row["created_at"]) or _now(),
        )

    # ----- data sources ------------------------------------------------------

    def create_data_source(self, name: str, config: DataConfig) -> DataSourceRecord:
        rec = DataSourceRecord(id=new_id(), name=name, config=config, created_at=_now())
        path = self.settings.data_sources_dir / f"{rec.id}.json"
        path.write_text(rec.model_dump_json(indent=2))
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO data_sources (id, name, created_at, config_json) VALUES (?,?,?,?)",
                (rec.id, rec.name, _iso(rec.created_at), config.model_dump_json()),
            )
        return rec

    def list_data_sources(self) -> list[DataSourceRecord]:
        with self._cursor() as cur:
            rows = cur.execute("SELECT * FROM data_sources ORDER BY created_at DESC").fetchall()
        return [self._row_to_data_source(row) for row in rows]

    def _row_to_data_source(self, row: sqlite3.Row) -> DataSourceRecord:
        return DataSourceRecord(
            id=row["id"],
            name=row["name"],
            config=DataConfig.model_validate_json(row["config_json"]),
            created_at=_parse_iso(row["created_at"]) or _now(),
        )

    # ----- runs --------------------------------------------------------------

    def create_run(self, spec: RunSpec, name: str | None = None) -> RunRecord:
        run_id = new_id()
        run_name = name or f"{spec.kind} {run_id[:6]}"
        rec = RunRecord(
            id=run_id,
            kind=spec.kind,
            status="queued",
            name=run_name,
            started_at=_now(),
            agent_id=spec.agent_id,
            env_id=spec.env_id,
            parent_run_id=spec.parent_run_id,
            spec=spec,
        )
        run_dir = self.settings.runs_dir / run_id
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(spec.model_dump_json(indent=2))
        (run_dir / "events.jsonl").touch()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO runs (id, kind, status, name, started_at, agent_id, env_id,"
                " parent_run_id, spec_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    rec.id,
                    rec.kind,
                    rec.status,
                    rec.name,
                    _iso(rec.started_at),
                    rec.agent_id,
                    rec.env_id,
                    rec.parent_run_id,
                    spec.model_dump_json(),
                ),
            )
        return rec

    def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        pid: int | None = None,
        exit_code: int | None = None,
        notes: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        with self._cursor() as cur:
            current = cur.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if current is None:
                return
            ended_at = current["ended_at"]
            if status in ("complete", "stopped", "failed") and ended_at is None:
                ended_at = _iso(_now())
            summary_json = (
                json.dumps(summary) if summary is not None else current["summary_json"]
            )
            cur.execute(
                "UPDATE runs SET status = ?, pid = COALESCE(?, pid), exit_code = COALESCE(?, exit_code),"
                " ended_at = ?, notes = COALESCE(?, notes), summary_json = ?"
                " WHERE id = ?",
                (status, pid, exit_code, ended_at, notes, summary_json, run_id),
            )

    def list_runs(self, *, kind: str | None = None, status: str | None = None) -> list[RunRecord]:
        query = "SELECT * FROM runs"
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY started_at DESC"
        with self._cursor() as cur:
            rows = cur.execute(query, params).fetchall()
        return [self._row_to_run(row) for row in rows]

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._cursor() as cur:
            row = cur.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._row_to_run(row) if row else None

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        spec = RunSpec.model_validate_json(row["spec_json"]) if row["spec_json"] else None
        summary = json.loads(row["summary_json"] or "{}")
        return RunRecord(
            id=row["id"],
            kind=row["kind"],
            status=row["status"],
            name=row["name"],
            started_at=_parse_iso(row["started_at"]) or _now(),
            ended_at=_parse_iso(row["ended_at"]),
            agent_id=row["agent_id"],
            env_id=row["env_id"],
            parent_run_id=row["parent_run_id"],
            pid=row["pid"],
            exit_code=row["exit_code"],
            notes=row["notes"] or "",
            spec=spec,
            summary=summary,
        )

    def run_dir(self, run_id: str) -> Path:
        return self.settings.runs_dir / run_id

    # ----- checkpoints -------------------------------------------------------

    def record_checkpoint(
        self,
        *,
        run_id: str | None,
        agent_id: str | None,
        step: int,
        path: str,
        metric_name: str | None = None,
        metric_value: float | None = None,
        tag: str | None = None,
    ) -> CheckpointRecord:
        rec = CheckpointRecord(
            id=new_id(),
            run_id=run_id,
            agent_id=agent_id,
            step=step,
            metric_name=metric_name,
            metric_value=metric_value,
            tag=tag,
            path=path,
            created_at=_now(),
        )
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO checkpoints (id, run_id, agent_id, step, metric_name,"
                " metric_value, tag, path, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    rec.id,
                    rec.run_id,
                    rec.agent_id,
                    rec.step,
                    rec.metric_name,
                    rec.metric_value,
                    rec.tag,
                    rec.path,
                    _iso(rec.created_at),
                ),
            )
        return rec

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        with self._cursor() as cur:
            row = cur.execute("SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)).fetchone()
        return self._row_to_checkpoint(row) if row else None

    def set_checkpoint_tag(self, checkpoint_id: str, tag: str | None) -> bool:
        with self._cursor() as cur:
            cur.execute("UPDATE checkpoints SET tag = ? WHERE id = ?", (tag, checkpoint_id))
            return cur.rowcount > 0

    def list_checkpoints(
        self, *, run_id: str | None = None, agent_id: str | None = None
    ) -> list[CheckpointRecord]:
        query = "SELECT * FROM checkpoints"
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY step DESC, created_at DESC"
        with self._cursor() as cur:
            rows = cur.execute(query, params).fetchall()
        return [self._row_to_checkpoint(row) for row in rows]

    def _row_to_checkpoint(self, row: sqlite3.Row) -> CheckpointRecord:
        return CheckpointRecord(
            id=row["id"],
            run_id=row["run_id"],
            agent_id=row["agent_id"],
            step=row["step"],
            metric_name=row["metric_name"],
            metric_value=row["metric_value"],
            tag=row["tag"],
            path=row["path"],
            created_at=_parse_iso(row["created_at"]) or _now(),
        )

    # ----- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _rmtree(path: Path) -> None:
    """Best-effort recursive delete used when removing entity directories."""

    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            _rmtree(child)
        else:
            child.unlink(missing_ok=True)
    path.rmdir()


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def reset_store_for_tests(settings: Settings) -> Store:
    """Replace the process-global store. Used only from test fixtures."""

    global _store
    if _store is not None:
        _store.close()
    _store = Store(settings=settings)
    return _store


def iter_events(path: Path) -> Iterable[dict[str, Any]]:
    """Yield parsed events from an events.jsonl file (one JSON object per line)."""

    if not path.exists():
        return
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
