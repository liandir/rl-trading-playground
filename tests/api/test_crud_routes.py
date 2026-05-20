"""CRUD smoke tests for agents, envs, data sources."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_agent_create_list_get_delete(client: TestClient) -> None:
    body = {
        "name": "baseline",
        "config": {"agent_type": "ppo", "network_preset": "tiny"},
    }
    created = client.post("/agents", json=body).json()
    assert created["name"] == "baseline"
    agent_id = created["id"]
    assert client.get("/agents").json()[0]["id"] == agent_id
    fetched = client.get(f"/agents/{agent_id}").json()
    assert fetched["config"]["agent_type"] == "ppo"
    assert client.delete(f"/agents/{agent_id}").status_code == 204
    assert client.get(f"/agents/{agent_id}").status_code == 404


def test_env_create_list_get_delete(client: TestClient) -> None:
    body = {"name": "default", "config": {}}
    created = client.post("/envs", json=body).json()
    env_id = created["id"]
    assert client.get(f"/envs/{env_id}").json()["name"] == "default"
    assert client.delete(f"/envs/{env_id}").status_code == 204


def test_data_source_preview_synthetic(client: TestClient) -> None:
    preview = client.post("/data/preview", json={"source": "synthetic"}).json()
    assert preview["n_steps"] == 256
    assert preview["n_assets"] == 3
    assert len(preview["asset_names"]) == 3


def test_data_source_create(client: TestClient) -> None:
    body = {"name": "synth", "config": {"source": "synthetic"}}
    rec = client.post("/data/sources", json=body).json()
    assert rec["name"] == "synth"
    listed = client.get("/data/sources").json()
    assert any(s["id"] == rec["id"] for s in listed)
