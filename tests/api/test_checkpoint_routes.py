"""Tagging checkpoints and attaching them to agents."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.services.store import Store


def _seed(store: Store) -> tuple[str, str, str]:
    """Create one run with a checkpoint file on disk and one agent. Returns (run_id, ckpt_id, agent_id)."""

    from src.api.schemas.agent import AgentConfig
    from src.api.schemas.data import DataConfig
    from src.api.schemas.env import EnvironmentConfig
    from src.api.schemas.run import RunSpec
    from src.api.schemas.training import TrainingConfig

    spec = RunSpec(
        kind="training",
        data=DataConfig(source="synthetic"),
        env=EnvironmentConfig(),
        agent=AgentConfig(agent_type="ppo", network_preset="tiny"),
        training=TrainingConfig(n_episodes=1, max_steps=8, warm_up=1, update_interval=3),
    )
    run = store.create_run(spec)
    ckpt_path = store.run_dir(run.id) / "checkpoints" / "latest.ptm"
    ckpt_path.write_bytes(b"")
    ckpt = store.record_checkpoint(
        run_id=run.id,
        agent_id=None,
        step=42,
        path=str(ckpt_path),
        metric_name="avg_reward",
        metric_value=1.2,
        tag="latest",
    )
    agent = store.create_agent("baseline", AgentConfig(agent_type="ppo", network_preset="tiny"))
    return run.id, ckpt.id, agent.id


def test_tag_endpoint_updates_then_clears(store: Store, client: TestClient) -> None:
    _, ckpt_id, _ = _seed(store)

    tagged = client.post(f"/checkpoints/{ckpt_id}/tag", json={"tag": "best"}).json()
    assert tagged["tag"] == "best"

    cleared = client.post(f"/checkpoints/{ckpt_id}/tag", json={"tag": None}).json()
    assert cleared["tag"] is None

    missing = client.post("/checkpoints/does-not-exist/tag", json={"tag": "x"})
    assert missing.status_code == 404


def test_attach_endpoint_sets_agent_checkpoint(store: Store, client: TestClient) -> None:
    run_id, ckpt_id, agent_id = _seed(store)

    response = client.post(f"/checkpoints/{ckpt_id}/attach", json={"agent_id": agent_id})
    assert response.status_code == 200

    agent = client.get(f"/agents/{agent_id}").json()
    assert agent["checkpoint_path"] is not None
    assert Path(agent["checkpoint_path"]).name == "latest.ptm"
    assert agent["parent_run_id"] == run_id


def test_attach_rejects_missing_agent(store: Store, client: TestClient) -> None:
    _, ckpt_id, _ = _seed(store)
    response = client.post(f"/checkpoints/{ckpt_id}/attach", json={"agent_id": "does-not-exist"})
    assert response.status_code == 404


def test_attach_rejects_when_file_gone(store: Store, client: TestClient) -> None:
    _, ckpt_id, agent_id = _seed(store)
    ckpt = store.get_checkpoint(ckpt_id)
    assert ckpt is not None
    Path(ckpt.path).unlink()
    response = client.post(f"/checkpoints/{ckpt_id}/attach", json={"agent_id": agent_id})
    assert response.status_code == 400


def test_list_filters(store: Store, client: TestClient) -> None:
    run_id, ckpt_id, _ = _seed(store)
    rows = client.get("/checkpoints", params={"run_id": run_id}).json()
    assert any(row["id"] == ckpt_id for row in rows)
