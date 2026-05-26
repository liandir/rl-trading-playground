"""Network preset definitions used by the runner.

Lifted from the deleted ``src/experiment/services.py`` with the duplicated
trunk/MLP block factored out. Activations and recurrent activations are
returned as strings so the preset dict survives a JSON round-trip; the
factory in :mod:`rl_trading_playground.api.runner.build` converts them back to callables.
"""
from __future__ import annotations

from typing import Any


_TRUNK = {
    "hidden_dims_asset": [256],
    "d_model": 64,
    "hidden_dims_mem": [1024],
    "hidden_dims_ff": [1024],
    "d_mem": 512,
    "d_ff": 512,
    "combine_mode": "concat",
    "hidden_dims_actor": [512],
    "hidden_dims_value": [512],
}


def _common(env: Any) -> dict[str, Any]:
    """Shared constructor kwargs derived from the environment."""

    M = len(env.tau_p)
    return {
        "num_assets": env.N,
        "d_asset": 4 * M + 9,
        "d_global": 10,
        "action_dim": env.action_dim,
        "activation": "gelu",
        "recurrent_activation": "tanh",
        "recurrent_type": "simple",
    }


def network_preset(name: str, env: Any) -> dict[str, Any]:
    """Return the network-config dict for ``name`` matched to ``env``."""

    common = _common(env)
    if name == "tiny":
        return {
            "type": "flat_per_asset_action_value",
            **common,
            "hidden_dims_asset": [16],
            "d_model": 8,
            "hidden_dims_mem": [16],
            "hidden_dims_ff": [16],
            "d_mem": 8,
            "d_ff": 8,
            "combine_mode": "concat",
            "hidden_dims_actor": [16],
            "hidden_dims_value": [16],
        }
    if name == "flat_per_asset":
        return {"type": "flat_per_asset_action_value", **common, **_TRUNK}
    if name == "attention_memory":
        return {
            "type": "attention_memory_action_value",
            **common,
            **_TRUNK,
            "asset_embed_dim": 16,
            "num_heads": 4,
            "n_att_layers": 3,
            "use_layer_norm": False,
        }
    if name == "auxiliary_per_asset":
        return {"type": "auxiliary_per_asset_action_value", **common, **_TRUNK}
    if name == "flat_model":
        return {"type": "flat_per_asset_model", **common, **_TRUNK}
    if name == "attention_model":
        return {
            "type": "attention_memory_model",
            **common,
            **_TRUNK,
            "asset_embed_dim": 16,
            "num_heads": 4,
            "n_att_layers": 3,
            "use_layer_norm": False,
        }
    if name == "spatiotemporal_auxiliary":
        return {
            "type": "spatiotemporal_aux_per_asset_action_value",
            **common,
            **_TRUNK,
            "temporal_num_heads": 4,
            "asset_num_heads": 4,
            "n_temporal_layers": 2,
            "n_asset_layers": 1,
            "window_size": 64,
        }
    raise ValueError(f"Unknown network preset '{name}'.")
