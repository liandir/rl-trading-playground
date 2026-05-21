"""End-to-end runs route: POST /runs spawns subprocess, events arrive over WS."""
from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient


def _spec(max_steps: int = 12) -> dict:
    return {
        "kind": "training",
        "data": {"source": "synthetic"},
        "env": {},
        "agent": {"agent_type": "ppo", "network_preset": "tiny"},
        "training": {
            "n_episodes": 1,
            "batch_size": 2,
            "max_steps": max_steps,
            "warm_up": 2,
            "update_interval": 4,
            "burn_in_updates": 0,
            "n_updates": 1,
            "save_checkpoint": True,
        },
    }


def _wait_for_status(client: TestClient, run_id: str, *targets: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = client.get(f"/runs/{run_id}").json()
        if rec["status"] in targets:
            return rec
        time.sleep(0.25)
    raise AssertionError(f"run {run_id} did not reach {targets}; last status={rec['status']}")


def test_run_creates_subprocess_and_completes(client: TestClient) -> None:
    created = client.post("/runs", json={"spec": _spec()}).json()
    run_id = created["id"]
    assert created["status"] in ("queued", "running")
    final = _wait_for_status(client, run_id, "complete", "failed")
    assert final["status"] == "complete", final
    events = client.get(f"/runs/{run_id}/events").json()
    kinds = [e["kind"] for e in events]
    assert "run_started" in kinds
    assert "run_finished" in kinds


def test_websocket_streams_events(client: TestClient) -> None:
    created = client.post("/runs", json={"spec": _spec()}).json()
    run_id = created["id"]
    received: list[dict] = []
    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        try:
            while True:
                msg = ws.receive_text()
                received.append(json.loads(msg))
        except Exception:
            pass
    kinds = [e["kind"] for e in received]
    assert "run_started" in kinds
    assert "run_finished" in kinds


def test_validation_run_writes_artifact(client: TestClient) -> None:
    spec = {
        "kind": "validation",
        "data": {"source": "synthetic"},
        "env": {},
        "agent": {"agent_type": "ppo", "network_preset": "tiny"},
        "validation": {"start_index": 10, "length": 8},
    }
    created = client.post("/runs", json={"spec": spec}).json()
    run_id = created["id"]
    final = _wait_for_status(client, run_id, "complete", "failed")
    assert final["status"] == "complete", final
    assert final["summary"].get("steps") == 8

    artifact = client.get(f"/runs/{run_id}/artifact").json()
    assert "metrics" in artifact and "series" in artifact
    series = artifact["series"]
    assert series["portfolio_value"], "series should not be empty"
    assert len(series["portfolio_value"]) == 8
    assert "asset_names" in series

    missing = client.get(f"/runs/{run_id}/artifact?name=does-not-exist")
    assert missing.status_code == 404


def test_stop_running_run_marks_status(client: TestClient) -> None:
    spec = _spec(max_steps=2_000)
    created = client.post("/runs", json={"spec": spec}).json()
    run_id = created["id"]
    _wait_for_status(client, run_id, "running")
    time.sleep(0.5)
    resp = client.post(f"/runs/{run_id}/stop")
    assert resp.status_code == 200
    final = _wait_for_status(client, run_id, "stopped", "complete", "failed")
    assert final["status"] in ("stopped", "complete"), final
