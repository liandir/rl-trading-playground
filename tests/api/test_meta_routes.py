"""Smoke tests for /healthz, /version, /registry."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_version(client: TestClient) -> None:
    r = client.get("/version")
    body = r.json()
    assert body["name"] == "trading-studio"
    assert body["api"] == "v1"


def test_registry_lists_agent_and_network_choices(client: TestClient) -> None:
    body = client.get("/registry").json()
    agent_names = {a["name"] for a in body["agent_types"]}
    preset_names = {p["name"] for p in body["network_presets"]}
    assert {"ppo", "aac", "aaq"} <= agent_names
    assert {"tiny", "attention_memory", "flat_per_asset"} <= preset_names
    assert "longshort_hierarchical_leverage" in {e["name"] for e in body["environment_types"]}
    assert "synthetic" in body["data_sources"]
