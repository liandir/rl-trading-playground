# Leveraged Long/Short Environment

This document describes the leveraged long/short environments implemented in:

- [src/environment/generic/longshort_leverage.py](/c:/Users/leand/Desktop/multi-currency-trading/src/environment/generic/longshort_leverage.py)
- [src/environment/discrete/longshort_hierarchical_leverage.py](/c:/Users/leand/Desktop/multi-currency-trading/src/environment/discrete/longshort_hierarchical_leverage.py)

The main public classes are:

- `LeveragedMultiCurrencyEnv`
- `BatchedLeveragedMultiCurrencyEnv`
- `LongShortHierarchicalLeverageEnv`
- `BatchedLongShortHierarchicalLeverageEnv`

The leveraged env keeps the same long/short mechanics as the non-leveraged `longshort` env, but changes the monetary interpretation of an open trade:

- in the non-leveraged env, cash directly becomes position notional
- in the leveraged env, cash is posted as margin, and the position notional becomes `margin * max_leverage`

## Overview

The leveraged env models a fixed-leverage linear perpetual style account with:

- long and short positions
- fixed leverage cap per env instance
- maintenance-margin validation before opening
- forced liquidation after price updates if equity falls below maintenance
- hierarchical discrete actions for direction and size bucket

Important constructor arguments:

- `max_leverage: float = 2.0`
- `maintenance_margin_ratio: float | None = None`

If `maintenance_margin_ratio` is not supplied, it defaults to `0.5 / max_leverage`.

## State Space

The leveraged state is flattened as:

```text
[ globals(10) | asset_0(3M+5) | asset_1(3M+5) | ... | asset_{N-1}(3M+5) ]
```

Total dimension:

```text
state_dim = 3*M*N + 5*N + 10
```

Where:

- `N` = number of tradable assets
- `M` = number of smoothing horizons in `tau_p`

### Global Features

| Feature | Shape | Meaning | Real-world interpretation |
|---|---:|---|---|
| `time` | `6` | Cyclic time features | Time-of-day / week / year seasonality |
| `cash_rel` | `1` | `cash / V` | Free buying power as share of equity |
| `rho` | `1` | `committed_sum / (committed_sum + cash)` | How deployed the account is |
| `gross_leverage` | `1` | `gross_exposure / V` | Broker leverage meter |
| `margin_buffer` | `1` | `(V - maintenance_requirement) / V` | Distance to liquidation |

### Per-Asset Features

| Feature | Shape | Meaning | Real-world interpretation |
|---|---:|---|---|
| `p_rel` | `M` | Price relative to smoothed price baselines | Multi-horizon trend / momentum deviation |
| `v_rel` | `M` | Volume relative to smoothed volume baselines | Unusual activity vs baseline |
| `vol_rel` | `M` | High-low range relative to smoothed range baselines | Volatility regime |
| `hl_rel` | `1` | Current normalized high-low range | Candle range |
| `x_rel` | `1` | `(pos_units * price) / V` | Signed portfolio exposure weight |
| `c_rel` | `1` | Asset share of total posted margin | Collateral allocation |
| `unrl_rel` | `1` | Unrealized after-tax PnL divided by posted margin | Return on margin |
| `side_rel` | `1` | `-1`, `0`, `+1` | Short / flat / long flag |

### What Is New Compared To Non-Leveraged Long/Short

The leveraged state adds exactly two global features that do not exist in the non-leveraged env:

- `gross_leverage`
- `margin_buffer`

These are the two features that tell the policy how close the account is to a margin event.

## Action Space

The hierarchical leveraged env uses the same high-level action contract as `longshort_hierarchical`.

Action:

```text
(a_d, a_q)
```

Where:

- `a_d` = primary action head
- `a_q` = size bucket head

### Primary Action Head

For `N` assets:

| `a_d` range | Meaning |
|---|---|
| `0` | Hold |
| `1 .. N` | Open long on asset `k = a_d - 1` |
| `N+1 .. 2N` | Open short on asset `k = a_d - 1 - N` |
| `2N+1 .. 3N` | Close asset `k = a_d - 1 - 2N` |

### Bucket Action Head

`a_q` selects an index into `size_buckets`.

Example:

```python
size_buckets = (0.10, 0.25, 0.50, 1.00)
```

Then:

- `a_q = 0` means use `10%` of available cash as margin budget
- `a_q = 1` means use `25%`
- `a_q = 2` means use `50%`
- `a_q = 3` means use `100%`

For hold and close actions, `a_q` is ignored.

### Monetary Meaning Of An Open Trade

The leveraged env changes the interpretation of the size bucket.

In the non-leveraged env:

```text
budget = frac * cash
position_notional = budget - open_fee
units = position_notional / price
```

In the leveraged env:

```text
budget = frac * cash
margin = budget - open_fee
position_notional = margin * max_leverage
units = position_notional / price
```

So the bucket is not "how much position to buy directly". It is "how much cash to post as margin".

## Money Flow

### Opening A Position

Suppose:

- cash = `$1,000`
- `size_bucket = 0.50`
- `open_fee = $1`
- `max_leverage = 3`
- price = `$100`

Then:

```text
budget   = 0.50 * 1000 = 500
margin   = 500 - 1 = 499
notional = 499 * 3 = 1497
units    = 1497 / 100 = 14.97
```

Monetary flow:

1. The account spends `$500` cash.
2. `$1` is consumed as fee.
3. `$499` becomes posted margin (`committed`).
4. The position receives `$1,497` of market exposure.

In code terms, `committed` is margin, not gross exposure.

### While The Position Is Open

The key quantities are:

- `gross_exposure = sum(abs(pos_units) * price)`
- `gross_leverage = gross_exposure / V`
- `maintenance_requirement = maintenance_margin_ratio * gross_exposure`
- `margin_buffer = (V - maintenance_requirement) / V`

As price moves:

- unrealized PnL changes with full leveraged notional
- portfolio equity `V` changes
- maintenance requirement changes with gross exposure
- margin buffer shrinks or expands

### Closing A Position

When closing, the env:

1. computes realized PnL from current price vs entry
2. releases the relevant share of posted margin
3. subtracts close fee
4. applies tax to positive realized gains
5. returns the net proceeds to cash

For a profitable long:

```text
net_proceeds = released_margin + pnl - close_fee - tax_on_positive_gain
```

For a losing long:

```text
net_proceeds = max(released_margin + pnl - close_fee, 0)
```

Shorts use the same signed PnL logic through `units * (price - entry_price)`.

### Liquidation

After every market update, the env checks:

```text
V < maintenance_requirement
```

If true:

- all open positions are force-closed
- `liquidated=True` is reported in `info`
- the state after liquidation is flat

This is intentionally account-level liquidation, not a single-position partial reduction.

## Validity Rules

An action can be masked out even if it is syntactically valid.

Important cases:

- you cannot open more in the same direction on an already same-sided position
- a flip first needs the opposite side to be fully closable
- a new open is rejected if it would immediately violate maintenance margin

So the leverage env mask is not just "can I afford the fee?" It is also "would the broker allow this margin usage?"

## Code Snippets

### 1. Create A Scalar Leveraged Hierarchical Env

```python
import torch

from src.environment.discrete.longshort_hierarchical_leverage import (
    LongShortHierarchicalLeverageEnv,
)

tau_p = torch.tensor([60 * 30, 60 * 60 * 4, 60 * 60 * 24], dtype=torch.float32)

env = LongShortHierarchicalLeverageEnv(
    N=3,
    C0=1_000.0,
    tau_p=tau_p,
    size_buckets=(0.10, 0.25, 0.50, 1.00),
    max_leverage=3.0,
    maintenance_margin_ratio=None,   # defaults to 0.5 / max_leverage
    open_fee=1.0,
    close_fee=1.0,
    tax_rate=0.26,
    reward_mode="log",
    dtype=torch.float32,
)
```

### 2. Reset And Step With Hierarchical Actions

```python
market = {
    "time": 1714212000.0,
    "close": torch.tensor([100.0, 200.0, 50.0]),
    "high": torch.tensor([102.0, 202.0, 51.0]),
    "low": torch.tensor([99.0, 198.0, 49.5]),
    "volume": torch.tensor([1_000.0, 2_000.0, 1_500.0]),
}

state = env.reset(data=market)

# Open a long in asset 0 using bucket 2 -> size_buckets[2] = 0.50
next_state, reward, done, info = env.step((1, 2), data=market)

print("state_dim:", env.state_dim)
print("action_dim:", env.action_dim)
print("gross_leverage:", info["gross_leverage"])
print("margin_buffer:", info["margin_buffer"])
print("liquidated:", info["liquidated"])
```

### 3. Inspect The Hierarchical Mask

```python
mask = env.valid_action_mask()

print(mask["primary"].shape)  # (1 + 3N,)
print(mask["buy"].shape)      # (N, K)
print(mask["sell"].shape)     # (N, K)

# Example: valid long buckets for asset 0
print(mask["buy"][0])
```

### 4. Batched Training Env

```python
from src.environment.discrete.longshort_hierarchical_leverage import (
    BatchedLongShortHierarchicalLeverageEnv,
)

bat_env = BatchedLongShortHierarchicalLeverageEnv(
    B=8,
    N=3,
    C0=1_000.0,
    tau_p=tau_p,
    size_buckets=(0.10, 0.25, 0.50, 1.00),
    max_leverage=3.0,
    dtype=torch.float32,
)

print(bat_env.state_dim)
print(bat_env.action_dim)
```

## Kraken Futures API Analogue

This env is not wire-compatible with Kraken Futures. It is a conceptual analogue.

The goal is:

- use the env for learning
- map learned intents to exchange-side futures concepts later

Kraken Futures concepts were checked against the official docs on April 27, 2026.

Official docs:

- Futures intro: https://docs.kraken.com/api/docs/guides/futures-introduction
- Send order: https://docs.kraken.com/api/docs/futures-api/trading/send-order
- Set leverage settings: https://docs.kraken.com/api/docs/futures-api/trading/set-leverage-setting
- Get open positions: https://docs.kraken.com/api/docs/futures-api/trading/get-open-positions
- Balances websocket: https://docs.kraken.com/api/docs/futures-api/websocket/balances/
- Open positions websocket: https://docs.kraken.com/api/docs/futures-api/websocket/open_position/

### Mapping Table

| Env concept | Kraken Futures analogue | Notes |
|---|---|---|
| `asset_idx` | `instrument` / contract symbol | Example: `PI_XBTUSD` |
| `long` / `short` action | `buy` / `sell` order side | Same directional intent |
| `size_bucket` | Chosen margin budget | Not an API field by itself; your execution layer must convert bucket -> order size |
| `max_leverage` | Contract leverage preference / isolated leverage cap | Closest analogue is `/leveragepreferences` |
| `committed` | Initial margin allocated to the position | Exchange reports related fields such as `initial_margin` |
| `gross_exposure` | Total position size in USD notionals | Similar to `total_position_size` / open position value |
| `gross_leverage` | Effective leverage | Kraken docs expose `effective_leverage` |
| `margin_buffer` | Cushion above maintenance | Closest live analogue comes from `margin_equity` vs `maintenance_margin` |
| forced liquidation | Exchange liquidation / unwind event | Env uses immediate full flattening after breach |

### Important Difference

In the env, leverage is fixed per environment instance. In Kraken Futures, leverage is not simply "attached to the action" in the same way:

- account margin mode matters
- leverage preferences can be configured at the contract level
- exchange-side position size is ultimately expressed as order quantity, not "margin bucket index"

So a production execution layer must convert:

```text
(asset_idx, side, size_bucket, max_leverage)
```

into:

```text
(instrument, order_side, order_qty, leverage preference / margin mode)
```

### Env-to-Exchange Intent Translation

A reasonable mapping pipeline is:

1. Read account state from the exchange.
2. Convert exchange balances and positions into local features or local execution constraints.
3. Let the model choose `(a_d, a_q)`.
4. Convert the chosen bucket to a target margin budget in quote currency.
5. Convert margin budget to target notional using configured leverage.
6. Convert target notional to order quantity for the selected futures contract.
7. Submit a futures order.

Pseudo-code:

```python
def env_action_to_target_notional(
    available_cash: float,
    size_bucket: float,
    open_fee: float,
    max_leverage: float,
) -> float:
    budget = size_bucket * available_cash
    margin = budget - open_fee
    if margin <= 0.0:
        return 0.0
    return margin * max_leverage
```

### Example Kraken Futures Order Intent

This is conceptual pseudo-code. The repo currently includes a spot REST helper in
[src/kraken/rest.py](/c:/Users/leand/Desktop/multi-currency-trading/src/kraken/rest.py), but not a Futures client.

```python
import requests

instrument = "PI_XBTUSD"
side = "buy"          # env long -> buy, env short -> sell
target_notional = 1_500.0
mark_price = 100_000.0
order_qty = target_notional / mark_price

pseudo_payload = {
    "instrument": instrument,   # illustrative only
    "side": side,               # illustrative only
    "size": order_qty,          # illustrative only
    "order_type": "market",     # illustrative only
}

# POST https://futures.kraken.com/derivatives/api/v3/sendorder
# Authentication omitted here on purpose.
# Field names here are illustrative; use the official doc for the exact schema.
response = requests.post(
    "https://futures.kraken.com/derivatives/api/v3/sendorder",
    data=pseudo_payload,
    timeout=10.0,
)
print(response.status_code)
```

### Example Leverage Preference Intent

If you want the exchange configuration to mirror the env's fixed leverage cap, the closest exchange-side idea is setting leverage preferences for the contract.

```python
import requests

pseudo_payload = {
    "instrument": "PI_XBTUSD",      # illustrative only
    "max_leverage": 3,              # illustrative only
    "margin_mode": "isolated",      # illustrative only
}

# PUT https://futures.kraken.com/derivatives/api/v3/leveragepreferences
# Authentication omitted here on purpose.
# Field names here are illustrative; use the official doc for the exact schema.
response = requests.put(
    "https://futures.kraken.com/derivatives/api/v3/leveragepreferences",
    data=pseudo_payload,
    timeout=10.0,
)
print(response.status_code)
```

The exact field names and auth headers should be taken from the official Kraken Futures docs above.

### Example Risk-State Analogue From Kraken Feeds

The env's added leverage features map most closely to Kraken Futures balance and position telemetry:

- env `gross_leverage` ~= Kraken `effective_leverage`
- env `margin_buffer` ~= a function of exchange `margin_equity` and `maintenance_margin`
- env `committed` ~= exchange `initial_margin`
- env forced liquidation threshold ~= exchange `liquidation_threshold` / maintenance breach state

Example local reconstruction:

```python
def margin_buffer_from_exchange(margin_equity: float, maintenance_margin: float) -> float:
    if margin_equity <= 0.0:
        return -1.0
    return (margin_equity - maintenance_margin) / margin_equity
```

## Practical Integration Notes

- The env is best treated as a training simulator, not an exact exchange emulator.
- Exchange execution needs separate handling for:
  - contract specs
  - minimum order sizes
  - tick sizes
  - isolated vs cross margin mode
  - liquidation engine details
  - funding and fees
- The env's liquidation logic is deliberately simpler than a real exchange liquidation engine.

## Suggested Next Steps

- add a dedicated Kraken Futures client next to `src/kraken/rest.py`
- define an explicit translator from `(a_d, a_q)` to target exchange order quantity
- decide whether live execution should use isolated or cross margin
- decide whether the live system should reconcile exchange `effective_leverage` back into the model state
