"""Capture per-run provenance so runs can be reproduced or audited later."""
from __future__ import annotations

import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _package(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _cuda_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda_available": False, "device_count": 0}
    return {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_version": getattr(torch.version, "cuda", None),
    }


def capture(seed: int | None = None) -> dict[str, Any]:
    """Build a JSON-friendly snapshot of the runtime environment."""

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_dirty": _git(["status", "--porcelain"]) not in (None, ""),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "packages": {
            "fastapi": _package("fastapi"),
            "pydantic": _package("pydantic"),
            "torch": _package("torch"),
            "uvicorn": _package("uvicorn"),
        },
        "torch_runtime": _cuda_info(),
        "seed": seed,
    }


def write(path: Path, data: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
