"""Interval-compatibility check and training-sidecar persistence."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.schemas.agent import AgentConfig, AgentRecord
from src.api.schemas.data import DataConfig
from src.api.services.compat import check_interval, read_sidecar
from src.api.services.store import Store


def _stub_agent_with_sidecar(
    store: Store, *, interval: int | None, agent_id: str = "stub"
) -> AgentRecord:
    rec = store.create_agent("stub", AgentConfig(agent_type="ppo", network_preset="tiny"))
    agent_dir = store.settings.agents_dir / rec.id
    ckpt = agent_dir / "checkpoint.ptm"
    ckpt.write_bytes(b"\x00")  # placeholder; we only test the sidecar
    meta = {
        "agent_config": AgentConfig(agent_type="ppo", network_preset="tiny").model_dump(),
        "step": 10,
        "saved_at": "2026-05-21T11:00:00Z",
        "extra": {},
    }
    if interval is not None:
        meta["data_config"] = DataConfig(source="synthetic", interval=interval).model_dump()
    (ckpt.with_suffix(".meta.json")).write_text(json.dumps(meta))
    store.update_agent_checkpoint(rec.id, str(ckpt))
    return store.get_agent(rec.id)  # type: ignore[return-value]


def test_check_interval_passes_when_match(store: Store) -> None:
    agent = _stub_agent_with_sidecar(store, interval=5)
    result = check_interval(agent, new_interval=5)
    assert result.ok
    assert result.training_interval == 5


def test_check_interval_fails_when_mismatch(store: Store) -> None:
    agent = _stub_agent_with_sidecar(store, interval=5)
    result = check_interval(agent, new_interval=15)
    assert not result.ok
    assert result.training_interval == 5
    assert "5-minute" in result.message and "15-minute" in result.message


def test_check_interval_unknown_when_no_sidecar_interval(store: Store) -> None:
    agent = _stub_agent_with_sidecar(store, interval=None)
    result = check_interval(agent, new_interval=5)
    assert result.ok and result.is_unknown
    assert "no recorded training interval" in result.message


def test_check_interval_handles_missing_sidecar(store: Store) -> None:
    rec = store.create_agent("bare", AgentConfig(agent_type="ppo", network_preset="tiny"))
    # No checkpoint attached at all.
    fresh = store.get_agent(rec.id)
    result = check_interval(fresh, new_interval=5)
    assert result.ok and result.is_unknown


def test_training_subprocess_writes_data_config_into_sidecar(
    client: TestClient, store: Store, tmp_path: Path
) -> None:
    spec = {
        "kind": "training",
        "data": {"source": "synthetic", "interval": 5},
        "env": {},
        "agent": {"agent_type": "ppo", "network_preset": "tiny"},
        "training": {
            "n_episodes": 1,
            "batch_size": 2,
            "max_steps": 8,
            "warm_up": 1,
            "update_interval": 3,
            "burn_in_updates": 0,
            "n_updates": 1,
            "save_checkpoint": True,
        },
    }
    created = client.post("/runs", json={"spec": spec}).json()
    run_id = created["id"]
    # Wait for terminal status.
    for _ in range(300):
        rec = client.get(f"/runs/{run_id}").json()
        if rec["status"] in ("complete", "stopped", "failed"):
            break
        import time

        time.sleep(0.1)
    else:
        raise AssertionError("training run never completed")
    assert rec["status"] == "complete", rec
    sidecar = store.run_dir(run_id) / "checkpoints" / "latest.meta.json"
    assert sidecar.exists()
    meta = read_sidecar(store.run_dir(run_id) / "checkpoints" / "latest.ptm")
    assert meta is not None
    assert meta.data_config is not None
    assert meta.data_config.interval == 5
