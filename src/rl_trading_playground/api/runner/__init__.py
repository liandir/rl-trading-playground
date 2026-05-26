"""Subprocess training/validation runner.

Spawned as `python -m rl_trading_playground.api.runner <run_id>` from the FastAPI process.
Reads `<store>/runs/<run_id>/config.json`, dispatches to training or
validation, and writes streaming events to `events.jsonl`.
"""
