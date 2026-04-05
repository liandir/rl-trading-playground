from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

import torch

from src.environment.live import LiveFrameSource


def _extract_action(agent_output):
    if isinstance(agent_output, tuple):
        return agent_output[0]
    return agent_output


def _call_act(agent, state: torch.Tensor, *, explore: bool):
    try:
        return _extract_action(agent.act(state, explore=explore))
    except TypeError:
        return _extract_action(agent.act(state))


def _default_store_transition(
    agent,
    state: torch.Tensor,
    action,
    next_state: torch.Tensor,
    reward: float,
    done: bool,
) -> None:
    if hasattr(agent, "buffer") and hasattr(agent.buffer, "store"):
        agent.buffer.store(
            state.detach().cpu(),
            action.detach().cpu() if isinstance(action, torch.Tensor) else torch.as_tensor(action),
            next_state.detach().cpu(),
            torch.tensor(float(reward)),
            torch.tensor(float(done)),
        )
        return

    if hasattr(agent, "store"):
        agent.store(state, action, next_state, reward, done)


def _default_update_agent(agent, *, batch_size: int) -> dict[str, float] | None:
    if not hasattr(agent, "update"):
        return None

    signature = inspect.signature(agent.update)
    if "batch_size" in signature.parameters:
        return agent.update(batch_size=batch_size)

    try:
        return agent.update(batch_size)
    except TypeError:
        return agent.update()


async def train_on_live_feed(
    agent,
    env,
    source: LiveFrameSource,
    *,
    n_steps: int | None = None,
    warm_up: int = 0,
    batch_size: int = 32,
    update_interval: int = 128,
    n_updates: int = 1,
    explore: bool = True,
    store_transition: Callable[..., None] | None = None,
    update_agent: Callable[..., dict[str, float] | None] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    store_transition = store_transition or _default_store_transition
    update_agent = update_agent or _default_update_agent

    history: list[dict[str, Any]] = []

    first_payload = await source.reset()
    state_obj = env.reset(first_payload)
    state = state_obj.to_tensor()

    step_idx = 0
    while n_steps is None or step_idx < n_steps:
        action = None
        if step_idx >= warm_up:
            action = _call_act(agent, state, explore=explore)

        payload = await source.step()
        next_state_obj, reward, done, info = env.step(action, payload)
        next_state = next_state_obj.to_tensor()

        if action is not None:
            store_transition(agent, state, action, next_state, reward, done)

        losses = None
        if action is not None and (step_idx + 1) % update_interval == 0:
            updates: list[dict[str, float]] = []
            for _ in range(max(int(n_updates), 1)):
                result = update_agent(agent, batch_size=batch_size)
                if result:
                    updates.append(result)
            if updates:
                losses = {
                    key: sum(item[key] for item in updates) / len(updates)
                    for key in updates[0]
                }

        record = {
            "step": step_idx,
            "reward": float(reward),
            "done": bool(done),
            "info": info,
            "loss": losses,
        }
        history.append(record)
        if on_step is not None:
            on_step(record)

        if done:
            state_obj = env.reset(payload)
            state = state_obj.to_tensor()
        else:
            state = next_state

        step_idx += 1

    return history


def train_on_live_feed_sync(*args, **kwargs):
    return asyncio.run(train_on_live_feed(*args, **kwargs))
