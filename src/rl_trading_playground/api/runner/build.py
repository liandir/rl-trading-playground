"""Factories that turn studio configs into torch objects.

Re-implements the parts of the deleted ``src/experiment/services.py`` that
mattered: market-data loading (prepared, CSV, synthetic), single + batched
environment construction, and agent construction from a preset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rl_trading_playground.agent.aac_hierarchical import HierarchicalAACAgent
from rl_trading_playground.agent.aac_hierarchical_aux import HierarchicalAuxAACAgent
from rl_trading_playground.agent.aac_hierarchical_model import HierarchicalModelAACAgent
from rl_trading_playground.agent.aaq_hierarchical import HierarchicalAAQAgent
from rl_trading_playground.agent.aaq_hierarchical_aux import HierarchicalAuxAAQAgent
from rl_trading_playground.agent.ppo_hierarchical import HierarchicalPPOAgent
from rl_trading_playground.agent.ppo_hierarchical_aux import HierarchicalAuxPPOAgent
from rl_trading_playground.agent.spatiotemporal_aac import SpatiotemporalHierarchicalAACAgent
from rl_trading_playground.api.runner.presets import network_preset
from rl_trading_playground.api.schemas.agent import AgentConfig
from rl_trading_playground.api.schemas.data import DataConfig
from rl_trading_playground.api.schemas.env import EnvironmentConfig
from rl_trading_playground.data import PAIRS, get_field, load_and_align_data
from rl_trading_playground.environment.longshort_hierarchical_leverage import (
    BatchedLongShortHierarchicalLeverageEnv,
    LongShortHierarchicalLeverageEnv,
)


AGENT_CLASSES = {
    "aac": HierarchicalAACAgent,
    "aaq": HierarchicalAAQAgent,
    "ppo": HierarchicalPPOAgent,
    "aac_aux": HierarchicalAuxAACAgent,
    "aaq_aux": HierarchicalAuxAAQAgent,
    "ppo_aux": HierarchicalAuxPPOAgent,
    "aac_model": HierarchicalModelAACAgent,
    "spatiotemporal_aac": SpatiotemporalHierarchicalAACAgent,
}


@dataclass
class MarketData:
    """Stacked OHLCV tensors used by historical envs."""

    open: torch.Tensor
    close: torch.Tensor
    high: torch.Tensor
    low: torch.Tensor
    volume: torch.Tensor
    times: torch.Tensor
    pairs: dict[str, str]

    @property
    def n_steps(self) -> int:
        return int(self.close.shape[0])

    @property
    def n_assets(self) -> int:
        return int(self.close.shape[1])


@dataclass
class EnvBundle:
    """Paired single + batched envs sharing the same parameterisation."""

    env: Any
    batched_env: Any


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_market_data(config: DataConfig) -> MarketData:
    """Load historical data following the same layout as the prior notebooks."""

    source = config.source
    prepared_path = Path(config.prepared_path)
    raw_base_path = Path(config.raw_base_path)
    if source == "auto":
        source = "prepared" if prepared_path.exists() else "kraken_csv"

    if source == "synthetic":
        return _synthetic_market_data()

    if source == "prepared":
        raw = torch.load(prepared_path, map_location="cpu", weights_only=False)
        times = raw["times"].to(dtype=torch.float64)
        close = raw["close"]
        high = raw.get("high", close)
        low = raw.get("low", close)
        open_ = raw.get("open", close)
        volume = raw["volume"]
        pairs = dict(raw.get("pairs", PAIRS))
    elif source == "kraken_csv":
        aligned, times_dt = load_and_align_data(
            PAIRS, base_path=str(raw_base_path), interval=config.interval
        )
        times = torch.tensor([t.timestamp() for t in times_dt], dtype=torch.float64)
        close = torch.tensor(get_field(aligned, "close"))
        high = torch.tensor(get_field(aligned, "high"))
        low = torch.tensor(get_field(aligned, "low"))
        open_ = torch.tensor(get_field(aligned, "open"))
        volume = torch.tensor(get_field(aligned, "volume"))
        if close.shape[0] == len(PAIRS):
            close, high, low, open_, volume = (x.T for x in (close, high, low, open_, volume))
        pairs = dict(PAIRS)
    else:
        raise ValueError(f"Unknown data source '{config.source}'.")

    sort_idx = torch.argsort(times)
    return MarketData(
        open=open_.to(dtype=torch.float32)[sort_idx],
        close=close.to(dtype=torch.float32)[sort_idx],
        high=high.to(dtype=torch.float32)[sort_idx],
        low=low.to(dtype=torch.float32)[sort_idx],
        volume=volume.to(dtype=torch.float32)[sort_idx],
        times=times[sort_idx],
        pairs=pairs,
    )


def _synthetic_market_data(T: int = 256, N: int = 3) -> MarketData:
    """Deterministic mini dataset for tests and offline UI work."""

    times = torch.arange(T, dtype=torch.float64) * 60.0
    base = torch.linspace(1.0, 1.2, T, dtype=torch.float32)[:, None]
    offsets = torch.arange(N, dtype=torch.float32)[None] * 0.1
    wave = 0.02 * torch.sin(torch.linspace(0, 12, T))[:, None]
    close = (base + offsets + wave).clamp_min(0.1)
    open_ = close * 0.999
    high = torch.maximum(open_, close) * 1.002
    low = torch.minimum(open_, close) * 0.998
    volume = torch.ones(T, N, dtype=torch.float32) * 1000.0
    pairs = {f"Asset {i + 1}": f"A{i + 1}" for i in range(N)}
    return MarketData(open=open_, close=close, high=high, low=low, volume=volume, times=times, pairs=pairs)


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def _dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype '{name}'.")


def build_environment(config: EnvironmentConfig, data: MarketData, *, batch_size: int = 8) -> EnvBundle:
    """Construct the single + batched leveraged hierarchical envs."""

    if config.env_type != "longshort_hierarchical_leverage":
        raise ValueError(f"Unsupported env_type '{config.env_type}'.")

    tau_p = torch.tensor([m * 60 for m in config.tau_minutes], dtype=_dtype(config.dtype))
    kwargs: dict[str, Any] = dict(
        N=data.n_assets,
        C0=config.cash,
        tau_p=tau_p,
        bankruptcy_threshold=config.bankruptcy_threshold,
        min_open_dollars=config.min_open_dollars,
        transaction_eps=config.transaction_eps,
        use_dollar_volume=config.use_dollar_volume,
        size_buckets=config.size_buckets,
        close_fee=config.close_fee,
        open_fee=config.open_fee,
        tax_rate=config.tax_rate,
        reward_mode=config.reward_mode,
        val_coeff=config.val_coeff,
        roi_coeff=config.roi_coeff,
        done_reward_penalty=config.done_reward_penalty,
        max_leverage=config.max_leverage,
        maintenance_margin_ratio=config.maintenance_margin_ratio,
        dtype=_dtype(config.dtype),
        device=config.device,
        eps=config.eps,
    )
    env = LongShortHierarchicalLeverageEnv(**kwargs, save_history=False)
    batched_env = BatchedLongShortHierarchicalLeverageEnv(B=batch_size, **kwargs)
    return EnvBundle(env=env, batched_env=batched_env)


# ---------------------------------------------------------------------------
# agent + network
# ---------------------------------------------------------------------------


_ACTIVATIONS: dict[str, Any] = {
    "gelu": torch.nn.GELU,
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "silu": torch.nn.SiLU,
    "swish": torch.nn.SiLU,
}

_RECURRENT_ACTIVATIONS: dict[str, Any] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
}


def _resolve_activation(value: Any) -> Any:
    if isinstance(value, str):
        cls = _ACTIVATIONS.get(value.lower().split(".")[-1])
        if cls is None:
            raise ValueError(f"Unsupported activation '{value}'.")
        return cls()
    return value


def _resolve_recurrent_activation(value: Any) -> Any:
    if isinstance(value, str):
        fn = _RECURRENT_ACTIVATIONS.get(value.lower().split(".")[-1])
        if fn is None:
            raise ValueError(f"Unsupported recurrent activation '{value}'.")
        return fn
    return value


def _resolve_network_config(network: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(network)
    if "activation" in cfg:
        cfg["activation"] = _resolve_activation(cfg["activation"])
    if "recurrent_activation" in cfg:
        cfg["recurrent_activation"] = _resolve_recurrent_activation(cfg["recurrent_activation"])
    return cfg


def build_agent(config: AgentConfig, env: Any) -> Any:
    """Construct an agent + its network from a studio :class:`AgentConfig`."""

    network = config.network_config or network_preset(config.network_preset, env)
    network = _resolve_network_config(network)
    cls = AGENT_CLASSES.get(config.agent_type)
    if cls is None:
        raise ValueError(f"Unknown agent_type '{config.agent_type}'.")
    kwargs: dict[str, Any] = dict(
        network=network,
        n_assets=env.N,
        n_buckets=env.K,
        gamma=config.gamma,
        vf_coef=config.vf_coef,
        ent_coef=config.ent_coef,
        normalize_advantages=config.normalize_advantages,
        advantage_type=config.advantage_type,
        gae_lambda=config.gae_lambda,
        dtype=env.dtype,
        device=config.device,
    )
    if config.agent_type.startswith("ppo"):
        kwargs["eps_clip"] = config.eps_clip
    if config.agent_type.endswith("_aux") or config.agent_type == "spatiotemporal_aac":
        kwargs["aux_coef"] = config.aux_coef
    if config.agent_type == "spatiotemporal_aac":
        kwargs["context_length"] = config.context_length
    return cls(**kwargs)
