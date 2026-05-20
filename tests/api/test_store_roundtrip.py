"""Store creates, reads, deletes for agents/envs/runs/checkpoints."""
from __future__ import annotations

from src.api.schemas.agent import AgentConfig
from src.api.schemas.data import DataConfig
from src.api.schemas.env import EnvironmentConfig
from src.api.schemas.run import RunSpec
from src.api.schemas.training import TrainingConfig
from src.api.services.store import Store


def test_agent_roundtrip(store: Store) -> None:
    rec = store.create_agent("baseline", AgentConfig(agent_type="ppo", network_preset="tiny"))
    assert store.get_agent(rec.id) is not None
    assert any(a.id == rec.id for a in store.list_agents())
    config_file = store.settings.agents_dir / rec.id / "config.json"
    assert config_file.exists()
    assert store.delete_agent(rec.id) is True
    assert store.get_agent(rec.id) is None


def test_env_roundtrip(store: Store) -> None:
    rec = store.create_env("default", EnvironmentConfig())
    assert store.get_env(rec.id) is not None
    assert store.delete_env(rec.id) is True
    assert store.get_env(rec.id) is None


def test_run_create_and_status_transitions(store: Store) -> None:
    spec = RunSpec(
        kind="training",
        data=DataConfig(source="synthetic"),
        env=EnvironmentConfig(),
        agent=AgentConfig(agent_type="ppo", network_preset="tiny"),
        training=TrainingConfig(n_episodes=1, max_steps=8, warm_up=1, update_interval=3),
    )
    run = store.create_run(spec)
    assert run.status == "queued"
    run_dir = store.run_dir(run.id)
    assert (run_dir / "config.json").exists()
    assert (run_dir / "events.jsonl").exists()
    store.update_run_status(run.id, "running", pid=42)
    refreshed = store.get_run(run.id)
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.pid == 42
    store.update_run_status(run.id, "complete", summary={"steps": 8})
    final = store.get_run(run.id)
    assert final is not None
    assert final.status == "complete"
    assert final.summary == {"steps": 8}
    assert final.ended_at is not None


def test_checkpoint_record_and_list(store: Store) -> None:
    spec = RunSpec(
        kind="training",
        data=DataConfig(source="synthetic"),
        env=EnvironmentConfig(),
        agent=AgentConfig(agent_type="ppo", network_preset="tiny"),
        training=TrainingConfig(n_episodes=1, max_steps=8, warm_up=1, update_interval=3),
    )
    run = store.create_run(spec)
    cp = store.record_checkpoint(
        run_id=run.id,
        agent_id=None,
        step=100,
        path=str(store.run_dir(run.id) / "checkpoints" / "step-100.ptm"),
        metric_name="total_reward",
        metric_value=1.5,
        tag="best",
    )
    listed = store.list_checkpoints(run_id=run.id)
    assert len(listed) == 1
    assert listed[0].id == cp.id
    assert listed[0].tag == "best"
