"""Adapters that paper over differences between PPO and actor-critic agents.

The agent classes have slightly different ``act``/``store``/``update``
signatures. Reflection-based dispatch belonged to the deleted training
module; here it lives in one place so the training loop reads cleanly.
"""
from __future__ import annotations

import inspect
from typing import Any

import torch

from src.agent.ppo_hierarchical import HierarchicalPPOAgent
from src.agent.utils import _build_aux_target


def is_ppo(agent: Any) -> bool:
    """True for any PPO-style agent (returns ``(action, log_prob)`` from ``act``)."""

    return isinstance(agent, HierarchicalPPOAgent) or agent.__class__.__name__.endswith("PPOAgent")


def store_accepts(agent: Any, name: str) -> bool:
    return name in inspect.signature(agent.store).parameters


def store_transition(
    agent: Any,
    *,
    obs: torch.Tensor,
    actions: torch.Tensor,
    log_probs: torch.Tensor | None,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    mask: Any,
    close: torch.Tensor,
    index: torch.Tensor,
) -> None:
    """Push one transition into the agent's rollout buffer."""

    extras: dict[str, Any] = {}
    if store_accepts(agent, "aux_target"):
        extras["aux_target"] = _build_aux_target(
            close,
            index,
            getattr(agent, "aux_horizons", (1, 5, 20)),
            reward=rewards,
        )
    args = [
        obs.detach(),
        actions.detach(),
    ]
    if is_ppo(agent):
        assert log_probs is not None
        args.append(log_probs.detach())
    args.extend([rewards.to(dtype=agent.dtype), dones.to(dtype=agent.dtype)])
    agent.store(*args, mask=mask, **extras)


def update_agent(
    agent: Any,
    *,
    next_obs: torch.Tensor,
    h0: Any,
    batched_env: Any,
    n_updates: int,
    max_grad_norm: float | None,
) -> dict[str, Any]:
    """Run one ``agent.update`` with whichever kwargs the agent supports."""

    params = inspect.signature(agent.update).parameters
    kwargs: dict[str, Any] = {"max_grad_norm": max_grad_norm}
    if "h0" in params:
        kwargs["h0"] = h0
    if "last_next_mask" in params:
        kwargs["last_next_mask"] = batched_env.valid_action_mask()
    if "k_epochs" in params:
        kwargs["k_epochs"] = n_updates
    return agent.update(next_obs, **kwargs)
