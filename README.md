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

- `generic/` — continuous-action long/short environments with and without
  leverage.
- `discrete/` — discrete-action variants including bucketed and hierarchical
  formulations.
- `hybrid/` — mixed continuous/discrete environments.
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
environment with an agent and network configuration. Configuration files for
batch experiments are kept in [configs/](configs/).

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
├── environment/  Trading environments (discrete, generic, hybrid, live)
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
