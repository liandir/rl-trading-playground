"""Historical batched training loop with streaming JSONL events.

Carries forward the batched rollout structure of the deleted
``src/experiment/training.py``, but:

* Emits one JSON line per event via :class:`EventSink` so the parent
  FastAPI process can tail progress.
* Checks ``terminal`` on every warm-up step (the prior version only checked
  on steps where it emitted an event, so a dones spike inside warm-up could
  keep stepping a terminated batched env).
* Reports any caller-provided ``should_stop`` predicate every step so the
  subprocess can be interrupted promptly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from src.agent.utils import _fmt_sim_elapsed, _sample_start_indices
from src.api.runner.adapters import is_ppo, store_transition, update_agent
from src.api.runner.build import EnvBundle, MarketData
from src.api.runner.events import EventSink
from src.api.schemas.agent import AgentConfig
from src.api.schemas.training import TrainingConfig


@dataclass
class TrainingResult:
    """Lightweight summary returned by :func:`run_training`."""

    n_episodes: int
    total_steps: int
    total_updates: int
    avg_reward: float
    final_portfolio: float
    stopped: bool


def run_training(
    agent: Any,
    env_bundle: EnvBundle,
    data: MarketData,
    config: TrainingConfig,
    *,
    sink: EventSink,
    agent_config: AgentConfig | None = None,
    checkpoint_path: Path | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TrainingResult:
    """Run an end-to-end batched training session, streaming events to ``sink``."""

    bat_env = env_bundle.batched_env
    if not hasattr(agent, "optim") or config.init_optimizer:
        agent.init_optimizer(config.lr, optim=config.optim)

    max_aux = max(tuple(getattr(agent, "aux_horizons", (0,))) or (0,))
    usable_T = data.close.shape[0] - int(max_aux)
    if usable_T <= config.max_steps:
        raise ValueError(
            f"Need more data than max_steps + aux horizon: T={usable_T} <= max_steps={config.max_steps}."
        )

    total_steps = 0
    total_updates = 0
    all_rewards: list[float] = []
    final_portfolio = 0.0
    stopped = False

    sink.emit("run_started", message=f"{config.n_episodes} episode(s)")

    for episode in range(1, config.n_episodes + 1):
        if should_stop is not None and should_stop():
            stopped = True
            break

        agent.net.reset(bat_env.B)
        h0 = agent.net.get_states(clone=True, detach=True)
        burn_in = int(config.burn_in_updates)
        starts = _sample_start_indices(usable_T, config.max_steps, batch_size=bat_env.B)
        sim_t0 = float(data.times[starts[0]])
        obs = bat_env.reset(
            data.open[starts],
            data.close[starts],
            data.high[starts],
            data.low[starts],
            data.volume[starts],
            data.times[starts],
        )
        agent.buffer.clear()
        ep_rewards: list[float] = []
        wall_t0 = time.monotonic()
        sink.emit("episode_start", episode=episode, message=f"episode {episode}")

        step = 1
        while step <= config.max_steps:
            if should_stop is not None and should_stop():
                stopped = True
                break

            si = starts + step
            in_warmup = step <= config.warm_up
            mask = bat_env.valid_action_mask() if not in_warmup else None

            if in_warmup:
                actions = None
                log_probs = None
                step_actions = (
                    torch.zeros(bat_env.B, dtype=torch.long),
                    torch.zeros(bat_env.B, dtype=torch.long),
                )
            else:
                act_result = agent.act(obs, mask=mask, explore=True, grad_enabled=False)
                if is_ppo(agent):
                    actions, log_probs = act_result
                else:
                    actions, log_probs = act_result, None
                step_actions = actions

            next_obs, rewards, dones = bat_env.step(
                step_actions,
                data.open[si],
                data.close[si],
                data.high[si],
                data.low[si],
                data.volume[si],
                data.times[si],
            )
            if actions is not None:
                ep_rewards.append(float(rewards.mean()))

            terminal = bool(dones.any()) or step >= config.max_steps

            if not in_warmup:
                store_transition(
                    agent,
                    obs=obs,
                    actions=actions,  # type: ignore[arg-type]
                    log_probs=log_probs,
                    rewards=rewards,
                    dones=dones,
                    mask=mask,
                    close=data.close,
                    index=si,
                )
                if (len(agent.buffer) >= config.update_interval or terminal) and len(agent.buffer) > 2:
                    h_t = agent.net.get_states(clone=True, detach=True)
                    if burn_in > 0:
                        burn_in -= 1
                    else:
                        loss_dict = update_agent(
                            agent,
                            next_obs=next_obs,
                            h0=h0,
                            batched_env=bat_env,
                            n_updates=config.n_updates,
                            max_grad_norm=config.max_grad_norm,
                        )
                        total_updates += 1
                        metrics = _metric_means(loss_dict)
                        sink.emit(
                            "update",
                            episode=episode,
                            step=step,
                            percent=100.0 * step / config.max_steps,
                            avg_reward=float(rewards.mean()),
                            avg_portfolio=float(bat_env.V.mean()),
                            sim_elapsed=_fmt_sim_elapsed(
                                float(data.times[starts[0] + step]) - sim_t0
                            ),
                            metrics=metrics,
                        )
                    agent.net.set_states(h_t, clone=True, detach=True)
                    h0 = h_t
                    agent.buffer.clear()
            else:
                if step % max(1, config.update_interval) == 1:
                    sink.emit(
                        "warm_up",
                        episode=episode,
                        step=step,
                        percent=100.0 * step / config.max_steps,
                        avg_portfolio=float(bat_env.V.mean()),
                        sim_elapsed=_fmt_sim_elapsed(
                            float(data.times[starts[0] + step]) - sim_t0
                        ),
                        message="warm-up",
                    )

            if terminal:
                break

            obs = next_obs
            step += 1

        total_steps += step
        all_rewards.extend(ep_rewards)
        final_portfolio = float(bat_env.V.mean())
        sink.emit(
            "episode_end",
            episode=episode,
            step=step,
            percent=min(100.0, 100.0 * step / config.max_steps),
            avg_reward=(sum(ep_rewards) / max(1, len(ep_rewards))) if ep_rewards else None,
            avg_portfolio=final_portfolio,
            sim_elapsed=f"{time.monotonic() - wall_t0:.1f}s wall",
            message=f"episode {episode} finished",
        )
        if stopped:
            break

    if config.save_checkpoint and checkpoint_path is not None and agent_config is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(checkpoint_path))
        sink.emit(
            "checkpoint_saved",
            step=total_steps,
            message=str(checkpoint_path),
            extra={"path": str(checkpoint_path)},
        )

    avg_reward = sum(all_rewards) / max(1, len(all_rewards))
    sink.emit(
        "run_stopped" if stopped else "run_finished",
        step=total_steps,
        percent=100.0,
        avg_reward=avg_reward,
        avg_portfolio=final_portfolio,
        message="training stopped" if stopped else "training complete",
    )
    return TrainingResult(
        n_episodes=config.n_episodes,
        total_steps=total_steps,
        total_updates=total_updates,
        avg_reward=avg_reward,
        final_portfolio=final_portfolio,
        stopped=stopped,
    )


def _metric_means(loss_dict: dict[str, list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, values in loss_dict.items():
        vals = [float(v) for v in values]
        out[key] = sum(vals) / max(1, len(vals))
    return out
