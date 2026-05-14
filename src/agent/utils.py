import torch
import torch.nn.functional as F


def apply_action_mask(logits: torch.Tensor, valid: torch.BoolTensor) -> torch.Tensor:
    """Set logits of invalid actions to -inf before sampling or loss computation."""
    return logits.masked_fill(~valid, float("-inf"))


def _fmt_sim_elapsed(elapsed_seconds: float) -> str:
    s = int(elapsed_seconds)
    d = s // 86400
    h = (s % 86400) // 3600
    return f"{d} day{'s' if d != 1 else ''} {h} h"


def _sample_start_indices(total_points: int, max_steps: int, batch_size: int = 1) -> torch.Tensor:
    """
    Sample valid reset indices for rollouts of ``max_steps`` environment steps.

    Reset consumes one data point and each environment step consumes one future
    point, so a full rollout requires at least ``max_steps + 1`` aligned market
    observations. Valid starts therefore lie in ``[0, total_points - max_steps)``.
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}.")
    if total_points <= max_steps:
        raise ValueError(
            f"Need at least max_steps + 1 data points to sample a rollout of {max_steps} steps, "
            f"got total_points={total_points}."
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    return torch.randint(total_points - max_steps, (batch_size,))


def _compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    v_t: torch.Tensor,
    v_tp1: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation.

    Returns ``(advantages, returns)`` where returns are used as critic targets.
    Inputs are expected to have leading time dimension first: ``(T, ...)``.
    """
    deltas = rewards + gamma * (1.0 - dones) * v_tp1 - v_t

    shape = deltas.shape
    T = shape[0]
    gae = torch.zeros(shape[1:], device=deltas.device, dtype=deltas.dtype)
    adv_list = []

    for t in reversed(range(T)):
        gae = deltas[t] + gamma * lam * (1.0 - dones[t]) * gae
        adv_list.append(gae)

    advantages = torch.stack(list(reversed(adv_list)), dim=0)
    returns = advantages + v_t
    return advantages, returns


def _compute_mc(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    v_t: torch.Tensor,
    v_tp1_last: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discounted Monte Carlo returns with bootstrap from the final next value."""
    T = rewards.shape[0]
    returns = torch.empty_like(rewards)
    g = v_tp1_last
    for t in reversed(range(T)):
        g = rewards[t] + gamma * (1.0 - dones[t]) * g
        returns[t] = g
    advantages = returns - v_t
    return advantages, returns


UPDATE_METRIC_NAMES = (
    "policy_objective",
    "value_loss",
    "entropy",
    "total_loss",
    "return_mean",
    "value_mean",
    "adv_abs_mean",
    "adv_pos_frac",
    "chosen_action_prob_mean",
    "policy_confidence_mean",
    "explained_variance",
)


def _empty_update_metrics() -> dict[str, list[float]]:
    return {name: [0.0] for name in UPDATE_METRIC_NAMES}


def _init_update_metrics() -> dict[str, list[float]]:
    return {name: [] for name in UPDATE_METRIC_NAMES}


def _explained_variance(targets: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    target_var = targets.var(unbiased=False)
    if float(target_var.detach().cpu().item()) <= 1e-12:
        return torch.zeros((), device=targets.device, dtype=targets.dtype)
    return 1.0 - residuals.var(unbiased=False) / (target_var + 1e-12)


def _append_update_metrics(
    metric_store: dict[str, list[float]],
    *,
    logits: torch.Tensor,
    log_probs: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    raw_advantages: torch.Tensor,
    policy_objective: torch.Tensor,
    value_loss: torch.Tensor,
    entropy: torch.Tensor,
    loss: torch.Tensor,
):
    probs = torch.softmax(logits.detach(), dim=-1)
    values_detached = values.detach()
    returns_detached = returns.detach()
    raw_advantages_detached = raw_advantages.detach()
    residuals = returns_detached - values_detached

    metrics = {
        "policy_objective": policy_objective.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "total_loss": loss.detach(),
        "return_mean": returns_detached.mean(),
        "value_mean": values_detached.mean(),
        "adv_abs_mean": raw_advantages_detached.abs().mean(),
        "adv_pos_frac": (raw_advantages_detached > 0).to(dtype=returns_detached.dtype).mean(),
        "chosen_action_prob_mean": log_probs.detach().exp().mean(),
        "policy_confidence_mean": probs.max(dim=-1).values.mean(),
        "explained_variance": _explained_variance(returns_detached, residuals),
    }
    for name, value in metrics.items():
        metric_store[name].append(float(value.cpu().item()))


def _gather_per_asset_mask(mask: torch.Tensor, asset_idx: torch.Tensor) -> torch.Tensor:
    """
    ``mask``: ``(..., N, K)`` bool, ``asset_idx``: ``(...)`` long.
    Returns the per-state slice ``mask[..., asset_idx, :]``.
    """
    expected_ndim = asset_idx.ndim + 2
    while mask.ndim < expected_ndim:
        mask = mask.unsqueeze(mask.ndim - 2)
    if mask.ndim != expected_ndim:
        raise ValueError(
            f"Expected mask ndim {expected_ndim} for asset_idx shape {tuple(asset_idx.shape)}, "
            f"got {mask.ndim}."
        )
    K = mask.shape[-1]
    idx = asset_idx.unsqueeze(-1).unsqueeze(-1).expand(*asset_idx.shape, 1, K)
    return mask.gather(-2, idx).squeeze(-2)


def _split_logits(logits: torch.Tensor, primary_dim: int, K: int):
    logits_d = logits[..., :primary_dim]
    logits_buy = logits[..., primary_dim:primary_dim + K]
    logits_sell = logits[..., primary_dim + K:primary_dim + 2 * K]
    return logits_d, logits_buy, logits_sell


def _resolve_historical_source(data, *, high=None, low=None, volume=None, times=None, open_=None):
    """Normalize historical inputs for single-environment training."""
    if isinstance(data, torch.Tensor):
        close = data
        missing = [
            name for name, value in (
                ("high", high),
                ("low", low),
                ("volume", volume),
                ("times", times),
                ("open_", open_),
            )
            if value is None
        ]
        if missing:
            raise KeyError(
                "Tensor historical input requires keyword tensors for "
                + ", ".join(missing)
                + "."
            )

        tensors = {
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "times": times,
            "open_": open_,
        }
        T = int(close.shape[0])
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
            if value.shape[0] != T:
                raise ValueError(f"{name} must have leading dimension {T}, got {value.shape[0]}.")

        def time_at(idx: int) -> float:
            return float(times[idx])

        def reset_env(env, idx: int):
            return env.reset(open_[idx], close[idx], high[idx], low[idx], volume[idx], times[idx])

        def step_env(env, action, idx: int):
            return env.step(action, open_[idx], close[idx], high[idx], low[idx], volume[idx], times[idx])

        return T, time_at, reset_env, step_env

    if any(value is not None for value in (high, low, volume, times)):
        raise ValueError("Pass either list-of-dicts historical data or raw tensors, not both.")

    T = len(data)

    def time_at(idx: int) -> float:
        return float(data[idx]["time"])

    def reset_env(env, idx: int):
        return env.reset(data[idx])

    def step_env(env, action, idx: int):
        return env.step(action, data=data[idx])

    return T, time_at, reset_env, step_env


def _unpack_policy_value_aux(output):
    if not isinstance(output, tuple) or len(output) != 3:
        raise ValueError(
            "Auxiliary hierarchical agents require a network returning "
            "(logits, critic, aux). Use network type 'auxiliary_per_asset_action_value'."
        )
    return output


def _close_from_historical(data):
    if isinstance(data, torch.Tensor):
        return data
    return torch.stack([step["close"] for step in data])


def _build_aux_target(close: torch.Tensor, index, horizons, *, reward=None, eps: float = 1e-12):
    device = close.device
    idx = torch.as_tensor(index, device=device, dtype=torch.long)
    squeeze = idx.dim() == 0
    if squeeze:
        idx = idx.unsqueeze(0)

    close_t = close.index_select(0, idx).clamp_min(eps)
    ret_targets = []
    vol_targets = []
    for horizon in horizons:
        h = int(horizon)
        future = close.index_select(0, idx + h).clamp_min(eps)
        ret_targets.append(torch.log(future / close_t))

        one_step = []
        for offset in range(1, h + 1):
            prev = close.index_select(0, idx + offset - 1).clamp_min(eps)
            nxt = close.index_select(0, idx + offset).clamp_min(eps)
            one_step.append(torch.log(nxt / prev))
        path_returns = torch.stack(one_step, dim=0)
        vol_targets.append(path_returns.pow(2).mean(dim=0).sqrt())

    target = {
        "asset_ret": torch.stack(ret_targets, dim=-1),
        "asset_vol": torch.stack(vol_targets, dim=-1),
    }
    if reward is not None:
        target["reward"] = torch.as_tensor(reward, device=device, dtype=close.dtype)
    if squeeze:
        target = {key: value.squeeze(0) for key, value in target.items()}
    return target


def _aux_loss(
    aux: dict[str, torch.Tensor],
    aux_targets: dict[str, torch.Tensor] | None,
    rewards: torch.Tensor,
    coefs: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = {}
    total = rewards.new_zeros(())

    reward_target = aux_targets.get("reward", rewards) if aux_targets is not None else rewards
    if "reward" in aux:
        losses["aux_reward_loss"] = F.smooth_l1_loss(aux["reward"], reward_target.detach())
        total = total + float(coefs.get("reward", 1.0)) * losses["aux_reward_loss"]

    if aux_targets is not None and "asset_ret" in aux and "asset_ret" in aux_targets:
        losses["aux_asset_ret_loss"] = F.smooth_l1_loss(aux["asset_ret"], aux_targets["asset_ret"].detach())
        total = total + float(coefs.get("asset_ret", 0.5)) * losses["aux_asset_ret_loss"]

    if aux_targets is not None and "asset_vol" in aux and "asset_vol" in aux_targets:
        losses["aux_asset_vol_loss"] = F.mse_loss(aux["asset_vol"], aux_targets["asset_vol"].detach())
        total = total + float(coefs.get("asset_vol", 0.25)) * losses["aux_asset_vol_loss"]

    losses["aux_loss"] = total
    return total, losses


def _policy_value_from_q(
    logits: torch.Tensor,
    q_values: torch.Tensor,
    valid_masks: torch.BoolTensor | None = None,
) -> torch.Tensor:
    if valid_masks is not None:
        logits = apply_action_mask(logits, valid_masks)
    probs = torch.softmax(logits, dim=-1)
    return (probs * q_values).sum(dim=-1)


def _q_taken(q_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return q_values.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)


def _next_valid_masks(valid_masks: torch.BoolTensor | None, action_dim: int) -> torch.BoolTensor | None:
    if valid_masks is None:
        return None
    last_mask = torch.ones_like(valid_masks[:1])
    return torch.cat([valid_masks[1:], last_mask], dim=0)


def _hierarchical_policy_value_from_q(
    logits: torch.Tensor,
    q_values: torch.Tensor,
    masks: dict | None,
    n_assets: int,
    n_buckets: int,
    primary_dim: int,
) -> torch.Tensor:
    """
    Policy expectation of the same additive factored Q used by
    ``_hierarchical_q_taken``.
    """
    expected = primary_dim + 2 * n_buckets
    if logits.shape[-1] != expected or q_values.shape[-1] != expected:
        raise ValueError(
            f"Expected logits/q_values last dim {expected}, got {logits.shape[-1]} and {q_values.shape[-1]}."
        )
    logits_d, logits_buy, logits_sell = _split_logits(logits, primary_dim, n_buckets)
    q_d, q_buy, q_sell = _split_logits(q_values, primary_dim, n_buckets)

    if masks is not None:
        logits_d = apply_action_mask(logits_d, masks["primary"])

    pi_d = torch.softmax(logits_d, dim=-1)
    v_d = (pi_d * q_d).sum(dim=-1)
    pi_buy_d = pi_d[..., 1:1 + n_assets]
    pi_sell_d = pi_d[..., 1 + n_assets:1 + 2 * n_assets]

    logits_buy_b = logits_buy.unsqueeze(-2).expand(*logits_buy.shape[:-1], n_assets, n_buckets)
    logits_sell_b = logits_sell.unsqueeze(-2).expand(*logits_sell.shape[:-1], n_assets, n_buckets)
    q_buy_b = q_buy.unsqueeze(-2).expand(*q_buy.shape[:-1], n_assets, n_buckets)
    q_sell_b = q_sell.unsqueeze(-2).expand(*q_sell.shape[:-1], n_assets, n_buckets)

    if masks is not None:
        buy_any = masks["buy"].any(dim=-1, keepdim=True)
        sell_any = masks["sell"].any(dim=-1, keepdim=True)
        buy_mask_safe = masks["buy"] | (~buy_any)
        sell_mask_safe = masks["sell"] | (~sell_any)
        logits_buy_b = apply_action_mask(logits_buy_b, buy_mask_safe)
        logits_sell_b = apply_action_mask(logits_sell_b, sell_mask_safe)

    v_buy = (torch.softmax(logits_buy_b, dim=-1) * q_buy_b).sum(dim=-1)
    v_sell = (torch.softmax(logits_sell_b, dim=-1) * q_sell_b).sum(dim=-1)
    return v_d + (pi_buy_d * v_buy).sum(dim=-1) + (pi_sell_d * v_sell).sum(dim=-1)


def _hierarchical_q_taken(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    n_assets: int,
    n_buckets: int,
    primary_dim: int,
) -> torch.Tensor:
    expected = primary_dim + 2 * n_buckets
    if q_values.shape[-1] != expected:
        raise ValueError(f"Expected q_values last dim {expected}, got {q_values.shape[-1]}.")
    if actions.shape[-1] != 2:
        raise ValueError(f"Expected hierarchical actions with last dim 2, got {tuple(actions.shape)}.")
    q_d, q_buy, q_sell = _split_logits(q_values, primary_dim, n_buckets)
    a_d = actions[..., 0].long()
    a_q = actions[..., 1].long()

    q_primary = q_d.gather(-1, a_d.unsqueeze(-1)).squeeze(-1)
    q_buy_taken = q_buy.gather(-1, a_q.unsqueeze(-1)).squeeze(-1)
    q_sell_taken = q_sell.gather(-1, a_q.unsqueeze(-1)).squeeze(-1)

    is_buy = (a_d >= 1) & (a_d <= n_assets)
    is_sell = (a_d >= n_assets + 1) & (a_d <= 2 * n_assets)
    return q_primary + is_buy.to(q_primary.dtype) * q_buy_taken + is_sell.to(q_primary.dtype) * q_sell_taken


def _check_same_shape(name: str, value: torch.Tensor, ref_name: str, ref: torch.Tensor) -> None:
    if value.shape != ref.shape:
        raise ValueError(f"{name}.shape={tuple(value.shape)} does not match {ref_name}.shape={tuple(ref.shape)}.")


def _next_masks(masks: dict | None, last_next_mask: dict | None = None) -> dict | None:
    if masks is None:
        return None
    if last_next_mask is None:
        return {key: torch.cat([value[1:], torch.ones_like(value[:1])], dim=0) for key, value in masks.items()}
    return {
        key: torch.cat([value[1:], last_next_mask[key].to(device=value.device, dtype=torch.bool).unsqueeze(0)], dim=0)
        for key, value in masks.items()
    }
