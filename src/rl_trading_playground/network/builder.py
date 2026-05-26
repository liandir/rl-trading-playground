"""Builder utilities for neural network architectures and reusable model components."""
import torch

from rl_trading_playground.network.action_q import ActionQNetwork
from rl_trading_playground.network.action_value import ActionValueNetwork
from rl_trading_playground.network.attention_memory_action_value import AttentionMemoryActionValueNetwork
from rl_trading_playground.network.attention_memory_model import AttentionMemoryModelNetwork
from rl_trading_playground.network.auxiliary_per_asset_action_value import AuxiliaryPerAssetActionValueNetwork
from rl_trading_playground.network.flat_per_asset_action_value import FlatPerAssetActionValueNetwork
from rl_trading_playground.network.flat_per_asset_model import FlatPerAssetModelNetwork
from rl_trading_playground.network.spatiotemporal_auxiliary_per_asset_action_value import SpatiotemporalAuxiliaryPerAssetActionValueNetwork


NETWORK_REGISTRY: dict[str, type[torch.nn.Module]] = {
    cls.name: cls
    for cls in (
        ActionValueNetwork,
        ActionQNetwork,
        AttentionMemoryActionValueNetwork,
        AttentionMemoryModelNetwork,
        FlatPerAssetActionValueNetwork,
        FlatPerAssetModelNetwork,
        AuxiliaryPerAssetActionValueNetwork,
        SpatiotemporalAuxiliaryPerAssetActionValueNetwork,
    )
}


def build_network(network: dict) -> torch.nn.Module:
    """Build a network from a config dict.

    The dict must contain a ``type`` key whose value matches the ``name``
    attribute of a registered network class. All other keys are forwarded as
    keyword arguments to the network constructor.

    Example
    -------
    >>> build_network({
    ...     "type": "action_value",
    ...     "state_dim": 62,
    ...     "action_dim": 37,
    ...     "hidden_dims": [128, 128],
    ... })

    Args:
        network (dict): The network value.

    Returns:
        torch.nn.Module: The computed or requested result.
    """
    cfg = dict(network)
    if "type" not in cfg:
        raise KeyError("network dict must contain a 'type' field.")
    type_name = cfg.pop("type")
    if type_name not in NETWORK_REGISTRY:
        raise ValueError(
            f"Unknown network type '{type_name}'. "
            f"Available: {sorted(NETWORK_REGISTRY)}."
        )
    return NETWORK_REGISTRY[type_name](**cfg)
