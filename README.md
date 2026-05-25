# RL Trading Playground

A research codebase for training and evaluating deep reinforcement learning
agents on multi-asset currency trading. The repository bundles trading
environments, neural network architectures, RL algorithms, live market
connectors, and the supporting tooling needed to run end-to-end experiments
from historical data to live execution.

## Repository Layout

```
src/
├── agent/        RL algorithms (AAC, AAQ, PPO and hierarchical / auxiliary / spatiotemporal variants)
├── environment/  Trading environments (discrete, generic, hybrid, live)
├── network/      Neural network building blocks and full architectures
├── kraken/       Kraken REST and WebSocket connectors for live data and order flow
├── news/         News collection, deduplication, and tagging pipeline
├── evaluate/     Visualisation and evaluation helpers
└── data.py       Dataset loading and preprocessing
configs/          Experiment configuration files
notebooks/        Training and analysis notebooks
scripts/          Utility scripts (live feeds, notebook tooling)
tests/            Test suite
```

## Components

### Environments (`src/environment/`)

Several flavours of multi-asset trading environments are provided. Each module
documents its own state space, action space, and reward modes in its docstring.

- `generic/` — continuous-action long/short environments with and without
  leverage.
- `discrete/` — discrete-action variants including bucketed and hierarchical
  formulations.
- `hybrid/` — mixed continuous/discrete environments.
- `live.py` — live execution wrapper backed by the Kraken connectors.

### Agents (`src/agent/`)

Implementations of advantage actor-critic (AAC), advantage actor-Q (AAQ), and
PPO, together with hierarchical, auxiliary-loss, model-based, and
spatiotemporal variants used in the training notebooks.

### Networks (`src/network/`)

Reusable neural network components in `core/` (attention, recurrent,
feedforward heads, per-asset branches) and full architectures that compose
them for action, value, action-value, and model heads, including
attention-memory and spatiotemporal variants.

### Market Connectors (`src/kraken/`)

REST client, WebSocket feed, message models, and resampling utilities for
Kraken market data and trading.

### News Pipeline (`src/news/`)

Configurable news collector with deduplication, tagging, and pluggable
sources. See [src/news/README.md](src/news/README.md) for details.

## Installation

The project is managed with [uv](https://github.com/astral-sh/uv) and requires
Python 3.11+.

```bash
uv sync
```

## Training and Evaluation

Training experiments live in [notebooks/](notebooks/). Each notebook pairs an
environment with an agent and network configuration. Configuration files for
batch experiments are kept in [configs/](configs/).

## Live Trading

The `scripts/` directory contains entry points for running live tickers and
order book streams against Kraken:

```bash
python3 scripts/live_ticker_stream.py
python3 scripts/live_book.py
```

Credentials are loaded from the environment; see `.env.example` for the
expected variables.

## Documentation

The documentation is published at
https://liandir.github.io/rl-trading-playground/.

Build the static documentation site with
[liandir/pydoc-builder](https://github.com/liandir/pydoc-builder):

```bash
uv run pydoc-builder --package src
```

The command reads Python docstrings without importing the project — optional
runtime dependencies and API credentials are not required to compile the docs
— and writes a GitHub Pages-compatible site to `docs/`. Open
`docs/index.html` locally or use the repository's `docs/` folder as the Pages
source.
