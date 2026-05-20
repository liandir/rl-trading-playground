"""End-to-end runner tests on synthetic data.

These tests import torch and instantiate real agents, so they are not the
fastest in the suite, but they validate that the deleted-experiment-layer
behaviour has been faithfully reconstructed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.api.runner.build import build_agent, build_environment, load_market_data
from src.api.runner.events import EventSink
from src.api.runner.training import run_training
from src.api.runner.validation import run_validation
from src.api.schemas.agent import AgentConfig
from src.api.schemas.data import DataConfig
from src.api.schemas.env import EnvironmentConfig
from src.api.schemas.run import RunSpec
from src.api.schemas.training import TrainingConfig
from src.api.schemas.validation import ValidationConfig
from src.api.services.store import Store


def test_run_training_emits_jsonl_events(tmp_path: Path) -> None:
    data = load_market_data(DataConfig(source="synthetic"))
    env_bundle = build_environment(EnvironmentConfig(), data, batch_size=2)
    agent = build_agent(AgentConfig(agent_type="ppo", network_preset="tiny"), env_bundle.env)
    events_path = tmp_path / "events.jsonl"
    with EventSink(events_path) as sink:
        result = run_training(
            agent,
            env_bundle,
            data,
            TrainingConfig(
                n_episodes=1,
                batch_size=2,
                max_steps=16,
                warm_up=2,
                update_interval=4,
                burn_in_updates=0,
                n_updates=1,
                save_checkpoint=False,
            ),
            sink=sink,
        )
    lines = events_path.read_text().splitlines()
    kinds = [json.loads(line)["kind"] for line in lines]
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_finished"
    assert "episode_end" in kinds
    assert "update" in kinds
    assert result.total_updates >= 1


def test_run_validation_produces_metrics_and_series(tmp_path: Path) -> None:
    data = load_market_data(DataConfig(source="synthetic"))
    env_bundle = build_environment(EnvironmentConfig(), data, batch_size=1)
    agent = build_agent(AgentConfig(agent_type="ppo", network_preset="tiny"), env_bundle.env)
    with EventSink(tmp_path / "events.jsonl") as sink:
        result = run_validation(
            agent,
            env_bundle.env,
            data,
            ValidationConfig(start_index=10, length=8),
            sink=sink,
        )
    assert result.metrics["steps"] == 8
    assert len(result.series["portfolio_value"]) == 8
    assert "asset_names" in result.series


def test_validation_restores_save_history_flag() -> None:
    data = load_market_data(DataConfig(source="synthetic"))
    env_bundle = build_environment(EnvironmentConfig(), data, batch_size=1)
    env_bundle.env.save_history = True
    agent = build_agent(AgentConfig(agent_type="ppo", network_preset="tiny"), env_bundle.env)
    run_validation(agent, env_bundle.env, data, ValidationConfig(start_index=10, length=4))
    assert env_bundle.env.save_history is True


def test_subprocess_entrypoint_runs_training_end_to_end(store: Store, tmp_path: Path) -> None:
    spec = RunSpec(
        kind="training",
        data=DataConfig(source="synthetic"),
        env=EnvironmentConfig(),
        agent=AgentConfig(agent_type="ppo", network_preset="tiny"),
        training=TrainingConfig(
            n_episodes=1,
            batch_size=2,
            max_steps=12,
            warm_up=2,
            update_interval=4,
            burn_in_updates=0,
            n_updates=1,
            save_checkpoint=True,
        ),
    )
    run = store.create_run(spec)
    env = os.environ.copy()
    env["STUDIO_STORE"] = str(store.settings.store_root)
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "src.api.runner", run.id],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    events_path = store.run_dir(run.id) / "events.jsonl"
    kinds = [json.loads(line)["kind"] for line in events_path.read_text().splitlines()]
    assert "run_started" in kinds
    assert "run_finished" in kinds
    refreshed = store.get_run(run.id)
    assert refreshed is not None
    assert refreshed.status == "complete"
    assert (store.run_dir(run.id) / "checkpoints" / "latest.ptm").exists()
    assert (store.run_dir(run.id) / "checkpoints" / "latest.meta.json").exists()
    assert (store.run_dir(run.id) / "provenance.json").exists()
