"""Subprocess entrypoint: ``python -m src.api.runner <run_id>``.

Reads the frozen run spec from ``<store>/runs/<run_id>/config.json``, builds
the data + environment + agent, and dispatches to training or validation
while writing progress events to ``events.jsonl``.
"""
from __future__ import annotations

import json
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.api.runner.build import build_agent, build_environment, load_market_data
from src.api.runner.events import EventSink
from src.api.runner.training import run_training
from src.api.runner.validation import run_validation
from src.api.schemas.checkpoint import CheckpointMeta
from src.api.schemas.run import RunSpec
from src.api.services import provenance
from src.api.services.store import Store, get_store
from src.api.settings import get_settings


_STOP_REQUESTED = False


def _request_stop(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _should_stop() -> bool:
    return _STOP_REQUESTED


def main(run_id: str) -> int:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    settings = get_settings()
    store: Store = get_store()
    run_dir: Path = settings.runs_dir / run_id
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"runner: missing config at {config_path}", file=sys.stderr)
        return 2
    spec = RunSpec.model_validate_json(config_path.read_text())
    seed = (
        spec.training.seed
        if spec.training is not None
        else (spec.validation.seed if spec.validation is not None else None)
    )
    provenance.write(run_dir / "provenance.json", provenance.capture(seed=seed))

    events_path = run_dir / "events.jsonl"
    store.update_run_status(run_id, "running")

    try:
        with EventSink(events_path) as sink:
            data = load_market_data(spec.data)
            batch_size = spec.training.batch_size if spec.training is not None else 1
            env_bundle = build_environment(spec.env, data, batch_size=batch_size)
            agent = build_agent(spec.agent, env_bundle.env)
            if spec.agent_id is not None:
                _load_existing_checkpoint(store, spec.agent_id, agent)

            if spec.kind == "training":
                if spec.training is None:
                    raise ValueError("training run requires training config")
                checkpoint_path = run_dir / "checkpoints" / "latest.ptm"
                result = run_training(
                    agent,
                    env_bundle,
                    data,
                    spec.training,
                    sink=sink,
                    agent_config=spec.agent,
                    checkpoint_path=checkpoint_path,
                    should_stop=_should_stop,
                )
                if spec.training.save_checkpoint and checkpoint_path.exists():
                    _write_checkpoint_meta(
                        run_dir=run_dir,
                        run_id=run_id,
                        spec=spec,
                        path=checkpoint_path,
                        step=result.total_steps,
                        metric_name="avg_reward",
                        metric_value=result.avg_reward,
                    )
                    store.record_checkpoint(
                        run_id=run_id,
                        agent_id=spec.agent_id,
                        step=result.total_steps,
                        path=str(checkpoint_path),
                        metric_name="avg_reward",
                        metric_value=result.avg_reward,
                        tag="latest",
                    )
                    if spec.agent_id is not None:
                        store.update_agent_checkpoint(
                            spec.agent_id, str(checkpoint_path), parent_run_id=run_id
                        )
                summary = {
                    "n_episodes": result.n_episodes,
                    "total_steps": result.total_steps,
                    "total_updates": result.total_updates,
                    "avg_reward": result.avg_reward,
                    "final_portfolio": result.final_portfolio,
                    "stopped": result.stopped,
                }
                store.update_run_status(
                    run_id,
                    "stopped" if result.stopped else "complete",
                    summary=summary,
                )
            elif spec.kind == "validation":
                if spec.validation is None:
                    raise ValueError("validation run requires validation config")
                result = run_validation(
                    agent, env_bundle.env, data, spec.validation, sink=sink
                )
                artifact_path = run_dir / "artifacts" / "validation.json"
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_text(
                    json.dumps({"metrics": result.metrics, "series": result.series}, default=str)
                )
                summary = {k: v for k, v in result.metrics.items() if isinstance(v, (int, float, bool))}
                store.update_run_status(run_id, "complete", summary=summary)
            else:
                raise ValueError(f"unknown run kind '{spec.kind}'")
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        try:
            with EventSink(events_path) as sink:
                sink.emit("run_failed", message=str(exc))
        except Exception:
            pass
        store.update_run_status(run_id, "failed", notes=str(exc)[:500])
        return 1
    return 0


def _load_existing_checkpoint(store: Store, agent_id: str, agent: Any) -> None:
    rec = store.get_agent(agent_id)
    if rec is None or not rec.checkpoint_path:
        return
    path = Path(rec.checkpoint_path)
    if path.exists():
        agent.load(str(path))


def _write_checkpoint_meta(
    *,
    run_dir: Path,
    run_id: str,
    spec: RunSpec,
    path: Path,
    step: int,
    metric_name: str,
    metric_value: float,
) -> None:
    meta = CheckpointMeta(
        agent_config=spec.agent,
        step=step,
        metric_name=metric_name,
        metric_value=metric_value,
        parent_run_id=run_id,
        saved_at=datetime.now(timezone.utc),
    )
    (path.with_suffix(".meta.json")).write_text(meta.model_dump_json(indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.api.runner <run_id>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
