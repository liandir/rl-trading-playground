"""Single-env validation rollout with metrics + chartable series."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch

from src.api.runner.build import MarketData
from src.api.runner.events import EventSink
from src.api.schemas.validation import ValidationConfig


@dataclass
class ValidationResult:
    metrics: dict[str, float | int | bool]
    series: dict[str, Any]


def run_validation(
    agent: Any,
    env: Any,
    data: MarketData,
    config: ValidationConfig,
    *,
    sink: EventSink | None = None,
) -> ValidationResult:
    """Run a deterministic or exploratory validation rollout."""

    start = config.start_index
    if start is None:
        start = int(data.n_steps * 0.9)
    start = max(0, min(int(start), data.n_steps - 2))
    end = min(data.n_steps - 1, start + int(config.length))
    if end <= start:
        raise ValueError("Validation range must include at least one step.")

    def make_data(idx: int) -> dict[str, Any]:
        return {
            "close": data.close[idx],
            "high": data.high[idx],
            "low": data.low[idx],
            "open": data.open[idx],
            "volume": data.volume[idx],
            "time": float(data.times[idx]),
        }

    agent.net.reset(1)
    prior_save_history = getattr(env, "save_history", None)
    if hasattr(env, "save_history"):
        env.save_history = False
    state = env.reset(make_data(start))
    hist_r: list[float] = []
    hist_i: list[dict[str, Any]] = []
    hist_a: list[Any] = []
    done = False
    if sink is not None:
        sink.emit("run_started", message=f"validation start={start} end={end}")

    try:
        total = end - start
        for offset, idx in enumerate(range(start + 1, end + 1), start=1):
            mask = env.valid_action_mask()
            act_result = agent.act(
                state.to_tensor(), mask=mask, explore=config.explore, grad_enabled=False
            )
            action = act_result[0] if isinstance(act_result, tuple) else act_result
            next_state, reward, done, info_t = env.step(action, data=make_data(idx))
            hist_r.append(float(reward))
            hist_i.append(info_t)
            hist_a.append(action.tolist() if hasattr(action, "tolist") else action)
            state = next_state
            if sink is not None and (offset % max(1, total // 50) == 0):
                sink.emit(
                    "validation_step",
                    step=offset,
                    percent=100.0 * offset / max(1, total),
                    avg_reward=float(reward),
                    avg_portfolio=float(info_t.get("V", 0.0)),
                )
            if done:
                break
    finally:
        if prior_save_history is not None and hasattr(env, "save_history"):
            env.save_history = prior_save_history

    if not hist_i:
        raise ValueError("Validation rollout produced no steps.")

    ts = [datetime.fromtimestamp(item["t"]) for item in hist_i]
    Vs = torch.tensor([item["V"] for item in hist_i], dtype=torch.float32)
    Cs = torch.tensor([item["C"] for item in hist_i], dtype=torch.float32)
    ps = torch.tensor([item["p"] for item in hist_i], dtype=torch.float32)
    pos_units = torch.tensor([item["pos_units"] for item in hist_i], dtype=torch.float32)
    committed = torch.tensor([item["committed"] for item in hist_i], dtype=torch.float32)
    rewards = torch.tensor(hist_r, dtype=torch.float32)
    action_types = torch.tensor([item["action_type"] for item in hist_i], dtype=torch.long)
    valid_trades = torch.tensor([float(item["valid_trade"]) for item in hist_i], dtype=torch.float32)
    realized_pnl = torch.tensor([sum(item["realized_pnl"]) for item in hist_i], dtype=torch.float32)
    realized_cost = torch.tensor([sum(item["realized_cost"]) for item in hist_i], dtype=torch.float32)

    portfolio_frac = pos_units * ps / Vs[:, None].clamp_min(1e-8)
    cash_frac = Cs / Vs.clamp_min(1e-8)
    cum_reward = rewards.cumsum(0)
    cum_realized_pnl = realized_pnl.cumsum(0)
    cum_realized_cost = realized_cost.cumsum(0)
    running_peak = torch.cummax(Vs, dim=0).values
    drawdown = 1.0 - Vs / running_peak.clamp_min(1e-8)
    norm_prices = ps / ps[0].clamp_min(1e-8)
    norm_value = Vs / Vs[0].clamp_min(1e-8)

    hold_mask = action_types == 0
    long_mask = action_types == 1
    short_mask = action_types == 2
    close_mask = action_types == 3
    trade_mask = long_mask | short_mask | close_mask
    invalid_mask = trade_mask & (valid_trades == 0)

    metrics: dict[str, float | int | bool] = {
        "steps": len(hist_i),
        "terminated": bool(done),
        "final_value": float(Vs[-1]),
        "final_cash": float(Cs[-1]),
        "total_return_pct": float((norm_value[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.max() * 100.0),
        "total_reward": float(cum_reward[-1]),
        "mean_reward": float(rewards.mean()),
        "reward_std": float(rewards.std(unbiased=False)),
        "long_count": int(long_mask.sum()),
        "short_count": int(short_mask.sum()),
        "close_count": int(close_mask.sum()),
        "hold_count": int(hold_mask.sum()),
        "invalid_trade_count": int(invalid_mask.sum()),
        "valid_trade_rate_pct": float(valid_trades.mean() * 100.0),
        "trade_rate_pct": float(trade_mask.to(torch.float32).mean() * 100.0),
        "realized_pnl_total": float(cum_realized_pnl[-1]),
        "realized_cost_total": float(cum_realized_cost[-1]),
    }
    series = {
        "timestamps": [t.isoformat() for t in ts],
        "asset_names": list(data.pairs.keys()),
        "portfolio_value": Vs.tolist(),
        "cash": Cs.tolist(),
        "prices": ps.tolist(),
        "normalized_prices": norm_prices.tolist(),
        "normalized_value": norm_value.tolist(),
        "rewards": rewards.tolist(),
        "cumulative_reward": cum_reward.tolist(),
        "drawdown": drawdown.tolist(),
        "portfolio_fraction": portfolio_frac.tolist(),
        "cash_fraction": cash_frac.tolist(),
        "committed": committed.tolist(),
        "cumulative_longs": long_mask.to(torch.float32).cumsum(0).tolist(),
        "cumulative_shorts": short_mask.to(torch.float32).cumsum(0).tolist(),
        "cumulative_closes": close_mask.to(torch.float32).cumsum(0).tolist(),
        "cumulative_invalid": invalid_mask.to(torch.float32).cumsum(0).tolist(),
        "cumulative_realized_pnl": cum_realized_pnl.tolist(),
        "cumulative_realized_cost": cum_realized_cost.tolist(),
        "actions": hist_a,
    }
    if sink is not None:
        sink.emit(
            "run_finished",
            step=len(hist_i),
            percent=100.0,
            avg_portfolio=float(Vs[-1]),
            message="validation complete",
            extra={"metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool))}},
        )
    return ValidationResult(metrics=metrics, series=series)
