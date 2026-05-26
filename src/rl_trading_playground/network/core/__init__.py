"""Package exports for rl_trading_playground.network.core, covering neural network architectures and reusable model components."""
from rl_trading_playground.network.core.attention import (
    ResidualCrossAttentionBlock,
    ResidualSelfAttentionBlock,
)
from rl_trading_playground.network.core.branches import LatentRecurrentFeedForwardBranch
from rl_trading_playground.network.core.feedforward_heads import (
    FeedForwardActionQNetwork,
    FeedForwardActionValueNetwork,
)
from rl_trading_playground.network.core.per_asset import BaseFlatPerAssetActionValueNetwork
from rl_trading_playground.network.core.recurrent import (
    RecurrentActionQNetwork,
    RecurrentActionValueNetwork,
    RecurrentNetwork,
    _build_recurrent_cell,
)
from rl_trading_playground.network.core.utils import (
    _build_causal_mask,
    _build_transformer_ffn,
    _init_linear,
    _merge_heads,
    _scaled_dot_product_attention,
    _sinusoidal_pos_encoding,
    _split_heads,
)
from rl_trading_playground.network.core.vanilla import (
    ActionValueNetwork,
    VanillaNetwork,
    VanillaQNetwork,
)

__all__ = [
    "ActionValueNetwork",
    "BaseFlatPerAssetActionValueNetwork",
    "FeedForwardActionQNetwork",
    "FeedForwardActionValueNetwork",
    "LatentRecurrentFeedForwardBranch",
    "RecurrentActionQNetwork",
    "RecurrentActionValueNetwork",
    "RecurrentNetwork",
    "ResidualCrossAttentionBlock",
    "ResidualSelfAttentionBlock",
    "VanillaNetwork",
    "VanillaQNetwork",
    "_build_causal_mask",
    "_build_recurrent_cell",
    "_build_transformer_ffn",
    "_init_linear",
    "_merge_heads",
    "_scaled_dot_product_attention",
    "_sinusoidal_pos_encoding",
    "_split_heads",
]
