"""Tests for /files/browse, /agents/inspect, /agents/import."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from rl_trading_playground.api.runner.build import build_agent, build_environment, load_market_data
from rl_trading_playground.api.schemas.agent import AgentConfig
from rl_trading_playground.api.schemas.data import DataConfig
from rl_trading_playground.api.schemas.env import EnvironmentConfig


def _make_ptm(tmp: Path) -> Path:
    data = load_market_data(DataConfig(source="synthetic"))
    bundle = build_environment(EnvironmentConfig(), data, batch_size=1)
    agent = build_agent(AgentConfig(agent_type="ppo", network_preset="tiny"), bundle.env)
    path = tmp / "imported.ptm"
    agent.save(str(path))
    return path


def test_files_browse_lists_ptm(client: TestClient, tmp_path: Path) -> None:
    ptm = _make_ptm(tmp_path)
    r = client.get(f"/files/browse?path={tmp_path}&ext=.ptm")
    assert r.status_code == 200
    body = r.json()
    names = [e["name"] for e in body["entries"]]
    assert ptm.name in names
    assert all(e["is_dir"] or e["name"].endswith(".ptm") for e in body["entries"])


def test_files_browse_accepts_multiple_extensions(client: TestClient, tmp_path: Path) -> None:
    ptm = _make_ptm(tmp_path)
    pt = tmp_path / "legacy.pt"
    pt.write_bytes(ptm.read_bytes())
    (tmp_path / "notes.txt").write_text("ignored")
    r = client.get(f"/files/browse?path={tmp_path}&ext=.ptm,.pt,.pth")
    assert r.status_code == 200
    names = [e["name"] for e in r.json()["entries"] if not e["is_dir"]]
    assert ptm.name in names
    assert pt.name in names
    assert "notes.txt" not in names


def test_files_browse_rejects_path_outside_root(client: TestClient) -> None:
    r = client.get("/files/browse?path=/etc")
    assert r.status_code in (400, 404)


def test_inspect_without_sidecar_returns_network_keys(client: TestClient, tmp_path: Path) -> None:
    ptm = _make_ptm(tmp_path)
    r = client.post("/agents/inspect", json={"path": str(ptm)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_sidecar"] is False
    assert body["network_keys"], "expected some network keys for a no-sidecar import"
    assert body["suggested"] is None


def test_inspect_with_sidecar_pre_fills_config(client: TestClient, tmp_path: Path) -> None:
    ptm = _make_ptm(tmp_path)
    sidecar = ptm.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "agent_config": AgentConfig(
                    agent_type="aac", network_preset="flat_per_asset", ent_coef=0.07
                ).model_dump(),
                "step": 1234,
                "metric_name": "avg_reward",
                "metric_value": 0.42,
                "parent_run_id": "prev-run-id",
                "saved_at": "2026-05-21T10:00:00Z",
                "extra": {},
            }
        )
    )
    r = client.post("/agents/inspect", json={"path": str(ptm)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_sidecar"] is True
    assert body["suggested"]["agent_type"] == "aac"
    assert body["suggested"]["network_preset"] == "flat_per_asset"
    assert body["suggested"]["ent_coef"] == 0.07
    assert body["parent_run_id"] == "prev-run-id"


def test_import_copies_ptm_and_records_checkpoint(client: TestClient, tmp_path: Path) -> None:
    ptm = _make_ptm(tmp_path)
    # Give it a sidecar so the response can carry the original config.
    sidecar = ptm.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "agent_config": AgentConfig(
                    agent_type="ppo", network_preset="tiny", ent_coef=0.02
                ).model_dump(),
                "step": 999,
                "metric_name": "avg_reward",
                "metric_value": -1.2,
                "parent_run_id": None,
                "saved_at": "2026-05-21T10:00:00Z",
                "extra": {},
            }
        )
    )
    edited = AgentConfig(agent_type="ppo", network_preset="tiny", ent_coef=0.11)
    r = client.post(
        "/agents/import",
        json={"name": "imported-1", "source_path": str(ptm), "config": edited.model_dump()},
    )
    assert r.status_code == 201, r.text
    agent = r.json()
    # File was copied into the agent's store dir.
    stored = Path(agent["checkpoint_path"])
    assert stored.exists()
    assert stored.parent.name == agent["id"]
    # Sidecar preserves both the edited config and the original.
    meta = json.loads((stored.parent / "checkpoint.meta.json").read_text())
    assert meta["agent_config"]["ent_coef"] == 0.11
    assert meta["extra"]["imported_from"]["agent_config"]["ent_coef"] == 0.02
    # A row was added to the checkpoints table tagged "imported".
    cps = client.get(f"/checkpoints?agent_id={agent['id']}").json()
    assert any(cp["tag"] == "imported" for cp in cps)


def test_import_preserves_training_data_config(client: TestClient, tmp_path: Path) -> None:
    ptm = _make_ptm(tmp_path)
    sidecar = ptm.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "agent_config": AgentConfig(agent_type="ppo", network_preset="tiny").model_dump(),
                "data_config": DataConfig(source="synthetic", interval=15).model_dump(),
                "env_config": EnvironmentConfig().model_dump(),
                "step": 10,
                "episode": 2,
                "metric_name": "avg_reward",
                "metric_value": 0.5,
                "parent_run_id": None,
                "saved_at": "2026-05-21T10:00:00Z",
                "extra": {},
            }
        )
    )
    r = client.post(
        "/agents/import",
        json={
            "name": "imported-2",
            "source_path": str(ptm),
            "config": AgentConfig(agent_type="ppo", network_preset="tiny").model_dump(),
        },
    )
    assert r.status_code == 201, r.text
    agent = r.json()
    meta = json.loads(
        (Path(agent["checkpoint_path"]).parent / "checkpoint.meta.json").read_text()
    )
    # Interval compatibility checks rely on data_config surviving the import.
    assert meta["data_config"]["interval"] == 15
    assert meta["env_config"] is not None
    assert meta["episode"] == 2


def test_import_rejects_non_ptm(client: TestClient, tmp_path: Path) -> None:
    txt = tmp_path / "garbage.txt"
    txt.write_text("not a checkpoint")
    r = client.post(
        "/agents/import",
        json={
            "name": "fail",
            "source_path": str(txt),
            "config": AgentConfig().model_dump(),
        },
    )
    assert r.status_code == 400
