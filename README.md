# RL Trading Playground

A research codebase for training and evaluating deep reinforcement learning
agents on multi-asset currency trading. The repository bundles trading
environments, neural network architectures, RL algorithms, live market
connectors, and the supporting tooling needed to run end-to-end experiments
from historical data to live execution.

## Components

### Environments (`src/rl_trading_playground/environment/`)

Several flavours of multi-asset trading environments are provided. Each module
documents its own state space, action space, and reward modes in its docstring.

- `discrete.py` — base discrete-action `MultiCurrencyEnv` and its batched
  counterpart, on which the variants below build.
- `generic/` — continuous-action long/short environments (`longshort.py`,
  `longshort_leverage.py`).
- `longshort*.py` — discrete-action long/short variants, including bucketed
  (`longshort_buckets.py`), hierarchical (`longshort_hierarchical.py`), and
  leveraged hierarchical (`longshort_hierarchical_leverage.py`) formulations.
- `discrete_buckets.py`, `discrete_log.py`, `discrete_log_buckets.py` — further
  discrete formulations with log-return and bucketed action spaces.
- `live.py` — live execution wrapper backed by the Kraken connectors.

### Agents (`src/rl_trading_playground/agent/`)

Implementations of advantage actor-critic (AAC), advantage actor-Q (AAQ), and
PPO, together with hierarchical, auxiliary-loss, model-based, and
spatiotemporal variants used in the training notebooks.

### Networks (`src/rl_trading_playground/network/`)

Reusable neural network components in `core/` (attention, recurrent,
feedforward heads, per-asset branches) and full architectures that compose
them for action, value, action-value, and model heads, including
attention-memory and spatiotemporal variants.

### Market Connectors (`src/rl_trading_playground/kraken/`)

REST client, WebSocket feed, message models, and resampling utilities for
Kraken market data and trading.

### News Pipeline (`src/rl_trading_playground/news/`)

Configurable news collector with deduplication, tagging, and pluggable
sources. See [src/rl_trading_playground/news/README.md](src/rl_trading_playground/news/README.md) for details.

## Installation

The project is managed with [uv](https://github.com/astral-sh/uv) and requires
Python 3.11+.

```bash
uv sync
```

The studio frontend is a separate Next.js app. The launcher installs its Node
dependencies automatically on first run if the local `next` executable is
missing. To pre-install them manually, run:

```bash
cd frontend
npm install
cd ..
```

## Quickstart

After `uv sync`, the fastest path to a trained agent is a `train_*` notebook in
[notebooks/](notebooks/). The minimal flow each one follows is:

```python
import torch
from rl_trading_playground.data import load_and_align_data, PAIRS, get_field
from rl_trading_playground.environment.longshort_hierarchical_leverage import (
    LongShortHierarchicalLeverageEnv,
    BatchedLongShortHierarchicalLeverageEnv,
)
from rl_trading_playground.agent.aac_hierarchical import HierarchicalAACAgent

# 1. Load and align historical OHLCV data (see "Data" below).
data, times = load_and_align_data(PAIRS, interval=5)
close = torch.tensor(get_field(data, "close")).T.float()      # (T, N)
# ...likewise for open / high / low / volume, and a (T,) times tensor.

# 2. Build a batched training environment and a single validation environment.
shared = dict(N=close.shape[1], C0=1_000, reward_mode="log", device="cpu")
bat_env = BatchedLongShortHierarchicalLeverageEnv(B=8, **shared)
env = LongShortHierarchicalLeverageEnv(**shared, save_history=False)

# 3. Build an agent and train on a train/validation split.
agent = HierarchicalAACAgent(network={"type": "attention_memory_action_value", ...})
T_train = int(close.shape[0] * 0.9)
agent.train_on_historical_bat(bat_env, open_t[:T_train], close[:T_train], ...,
                              n_episodes=3, lr=1e-5)
agent.save("data/agent/my_run.ptt")
```

> **Note:** the constructor and `train_on_historical_bat` arguments are
> abbreviated here — use a notebook for the full, working configuration.

### Data

The notebooks expect Kraken OHLCVT data under `data/`, in either of two forms:

- a preprocessed `data/historical_data0.ptt` tensor bundle (loaded directly), or
- raw Kraken CSVs under `data/Kraken_OHLCVT/` (downloadable from Kraken's
  [historical data export](https://support.kraken.com/hc/en-us/articles/360047124832)),
  which are loaded and aligned on the fly via `load_and_align_data`.

If neither is present the data-loading cell raises `FileNotFoundError`.

> ⚠️ **Live trading:** `environment/live.py` and the `kraken/` connectors can place
> real orders against a funded Kraken account. Keep API credentials out of the
> repo, start on paper/testnet, and treat any live run as financially risky.

## Running the Studio

The installed console command starts both the FastAPI backend and the Next.js
frontend in development mode:

```bash
uv run autotrading-playground
```

By default, the backend runs at http://127.0.0.1:8000 and the frontend runs at
http://127.0.0.1:3000. The launcher automatically sets:

- `NEXT_PUBLIC_API_URL` for the frontend.
- `STUDIO_CORS` for the backend.

Useful options:

```bash
uv run autotrading-playground \
  --backend-host 127.0.0.1 \
  --backend-port 8000 \
  --frontend-host 127.0.0.1 \
  --frontend-port 3000
```

If your frontend lives somewhere other than `frontend/`, pass its path:

```bash
uv run autotrading-playground --frontend-dir /path/to/frontend
```

Press `Ctrl+C` to stop both processes.

## Training and Evaluation

Training experiments live in [notebooks/](notebooks/). Each notebook pairs an
environment with an agent and network configuration. The `train_*` notebooks
load historical Kraken data, build a batched environment and agent, train via
`agent.train_on_historical_bat(...)`, and run a held-out validation rollout. The
[configs/](configs/) directory holds configuration for the news pipeline
(`news.toml.example`, `news_aliases.json`); experiment hyperparameters are set
inline in each notebook.

## Documentation

The documentation is published at
https://liandir.github.io/rl-trading-playground/.

Build the static documentation site with
[liandir/pydoc-builder](https://github.com/liandir/pydoc-builder):

```bash
uv run pydoc-builder --project-root src --package rl_trading_playground --docs-dir ../docs
```

The command reads Python docstrings without importing the project — optional
runtime dependencies and API credentials are not required to compile the docs
— and writes a GitHub Pages-compatible site to `docs/`. Open
`docs/index.html` locally or use the repository's `docs/` folder as the Pages
source.

## Repository Layout

```
src/rl_trading_playground/
├── agent/        RL algorithms (AAC, AAQ, PPO and hierarchical / auxiliary / spatiotemporal variants)
├── api/          FastAPI backend for the trading studio
├── data/         Dataset loading and preprocessing helpers
├── environment/  Trading environments (discrete base + long/short variants, generic, live)
├── network/      Neural network building blocks and full architectures
├── kraken/       Kraken REST and WebSocket connectors for live data and order flow
├── news/         News collection, deduplication, and tagging pipeline
└── evaluate/     Visualisation and evaluation helpers
frontend/         Next.js frontend for the trading studio
configs/          Experiment configuration files
notebooks/        Training and analysis notebooks
scripts/          Utility scripts (live feeds, notebook tooling)
tests/            Test suite
```
