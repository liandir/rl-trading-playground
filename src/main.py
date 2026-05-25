"""Command-line entrypoint for the trading studio."""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the FastAPI backend and Next.js frontend together.

    Args:
        argv: Optional command-line arguments. If omitted, arguments are read
            from ``sys.argv``.

    Returns:
        Process-style exit code. Returns ``0`` when the command is interrupted
        cleanly, ``2`` for local configuration errors, or the first child
        process exit code when a child exits unexpectedly.
    """
    args = _parse_args(argv)
    frontend_dir = args.frontend_dir.resolve()

    if not frontend_dir.is_dir():
        print(f"frontend directory not found: {frontend_dir}", file=sys.stderr)
        return 2
    if shutil.which("npm") is None:
        print("npm was not found on PATH; install Node.js/npm to run the frontend.", file=sys.stderr)
        return 2
    if not _ensure_frontend_dependencies(frontend_dir):
        return 2

    backend_url = f"http://{args.backend_host}:{args.backend_port}"
    frontend_origin = f"http://{args.frontend_host}:{args.frontend_port}"
    cors_origins = ",".join(
        dict.fromkeys(
            [
                frontend_origin,
                f"http://localhost:{args.frontend_port}",
                f"http://127.0.0.1:{args.frontend_port}",
            ]
        )
    )

    backend_env = os.environ.copy()
    backend_env["STUDIO_CORS"] = cors_origins

    frontend_env = os.environ.copy()
    frontend_env["NEXT_PUBLIC_API_URL"] = backend_url

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.app:app",
        "--host",
        args.backend_host,
        "--port",
        str(args.backend_port),
    ]
    frontend_cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--hostname",
        args.frontend_host,
        "--port",
        str(args.frontend_port),
    ]

    processes: list[subprocess.Popen[bytes]] = []
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        _terminate(processes)

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        print(f"Starting backend at {backend_url}", flush=True)
        backend = subprocess.Popen(backend_cmd, cwd=REPO_ROOT, env=backend_env)
        processes.append(backend)

        print(f"Starting frontend at {frontend_origin}", flush=True)
        frontend = subprocess.Popen(frontend_cmd, cwd=frontend_dir, env=frontend_env)
        processes.append(frontend)

        return _wait_for_exit(processes, lambda: stopping)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        _terminate(processes)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments for the studio launcher.

    Args:
        argv: Optional command-line arguments. If omitted, ``argparse`` reads
            from ``sys.argv``.

    Returns:
        Parsed launcher arguments.
    """
    parser = argparse.ArgumentParser(
        prog="autotrading-playground",
        description="Run the trading studio backend and frontend.",
    )
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-host", default="127.0.0.1")
    parser.add_argument("--frontend-port", type=int, default=3000)
    parser.add_argument("--frontend-dir", type=Path, default=REPO_ROOT / "frontend")
    return parser.parse_args(argv)


def _ensure_frontend_dependencies(frontend_dir: Path) -> bool:
    """Install frontend dependencies if the local Next.js binary is missing.

    Args:
        frontend_dir: Path to the frontend project directory.

    Returns:
        True when dependencies are already present or install successfully.
        False when the frontend package metadata is missing or installation
        fails.
    """
    if _has_frontend_dependencies(frontend_dir):
        return True

    if not (frontend_dir / "package.json").exists():
        print(f"frontend package.json not found: {frontend_dir / 'package.json'}", file=sys.stderr)
        return False

    print(f"Frontend dependencies are missing; running `npm install` in {frontend_dir}", flush=True)
    completed = subprocess.run(["npm", "install"], cwd=frontend_dir)
    if completed.returncode != 0:
        print(
            "frontend dependency installation failed; fix the npm error above "
            "and rerun `uv run autotrading-playground`.",
            file=sys.stderr,
        )
        return False

    if not _has_frontend_dependencies(frontend_dir):
        print(
            "frontend dependency installation completed, but the local Next.js "
            "executable is still missing.",
            file=sys.stderr,
        )
        return False

    return True


def _has_frontend_dependencies(frontend_dir: Path) -> bool:
    """Return whether the frontend has the local Next.js executable installed.

    Args:
        frontend_dir: Path to the frontend project directory.

    Returns:
        True if a platform-specific local Next.js executable exists in
        ``node_modules/.bin``.
    """
    bin_dir = frontend_dir / "node_modules" / ".bin"
    return (bin_dir / "next").exists() or (bin_dir / "next.cmd").exists()


def _wait_for_exit(
    processes: list[subprocess.Popen[bytes]],
    is_stopping: Callable[[], bool],
) -> int:
    """Wait until one managed subprocess exits.

    Args:
        processes: Subprocesses managed by the launcher.
        is_stopping: Callback that returns whether shutdown was requested by
            this process.

    Returns:
        ``0`` for a clean requested shutdown, otherwise the first exited child
        process return code.
    """
    while True:
        for process in processes:
            code = process.poll()
            if code is None:
                continue
            _terminate([p for p in processes if p is not process])
            return 0 if is_stopping() and code in (0, -signal.SIGTERM) else code
        time.sleep(0.2)


def _terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    """Terminate managed subprocesses, killing any that do not stop quickly.

    Args:
        processes: Subprocesses to terminate.
    """
    live = [process for process in processes if process.poll() is None]
    for process in live:
        process.terminate()

    deadline = time.monotonic() + 5
    for process in live:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass

    for process in live:
        if process.poll() is None:
            process.kill()
            process.wait()


if __name__ == "__main__":
    sys.exit(main())
