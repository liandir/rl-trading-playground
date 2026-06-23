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


REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the FastAPI backend and Next.js frontend together."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    frontend_dir = args.frontend_dir.resolve()

    if not frontend_dir.is_dir():
        return _error(f"frontend directory not found: {frontend_dir}")

    # Locate (installing if necessary) the local Next.js binary. The dev server
    # is launched from this binary directly rather than via `npm run dev`, so
    # the launcher is the single source of truth for the host and port.
    next_bin = _ensure_frontend_dependencies(frontend_dir, parser.prog)
    if next_bin is None:
        return 2

    backend_url = f"http://{args.backend_host}:{args.backend_port}"
    frontend_origin = f"http://{args.frontend_host}:{args.frontend_port}"

    backend_env = os.environ.copy()
    backend_env["STUDIO_CORS"] = _cors_origins(args.frontend_host, args.frontend_port)

    frontend_env = os.environ.copy()
    frontend_env["NEXT_PUBLIC_API_URL"] = backend_url

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "rl_trading_playground.api.app:app",
        "--host",
        args.backend_host,
        "--port",
        str(args.backend_port),
    ]
    frontend_cmd = [
        str(next_bin),
        "dev",
        "--hostname",
        args.frontend_host,
        "--port",
        str(args.frontend_port),
    ]

    return _run_studio(
        backend=(backend_cmd, REPO_ROOT, backend_env, f"Starting backend at {backend_url}"),
        frontend=(frontend_cmd, frontend_dir, frontend_env, f"Starting frontend at {frontend_origin}"),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the studio launcher."""
    parser = argparse.ArgumentParser(
        prog="autotrading-playground",
        description="Run the trading studio backend and frontend.",
    )
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-host", default="127.0.0.1")
    parser.add_argument("--frontend-port", type=int, default=3000)
    parser.add_argument("--frontend-dir", type=Path, default=REPO_ROOT / "frontend")
    return parser


def _cors_origins(frontend_host: str, frontend_port: int) -> str:
    """Comma-separated browser origins the backend should allow."""
    candidates = [
        f"http://{frontend_host}:{frontend_port}",
        f"http://localhost:{frontend_port}",
        f"http://127.0.0.1:{frontend_port}",
    ]
    # 0.0.0.0 is a bind address, not a valid browser origin — drop it when the
    # user opted into binding the frontend to all interfaces. dict.fromkeys
    # de-duplicates while preserving order.
    return ",".join(
        dict.fromkeys(o for o in candidates if "//0.0.0.0:" not in o)
    )


def _ensure_frontend_dependencies(frontend_dir: Path, prog: str) -> Path | None:
    """Return the local Next.js binary, running `npm install` first if missing.

    Returns the path to the binary on success, or ``None`` (after printing an
    explanation) on failure.
    """
    next_bin = _find_next_binary(frontend_dir)
    if next_bin is not None:
        return next_bin

    # npm is only required when dependencies actually need installing.
    if not (frontend_dir / "package.json").exists():
        _error(f"frontend package.json not found: {frontend_dir / 'package.json'}")
        return None
    npm = shutil.which("npm")
    if npm is None:
        _error("npm was not found on PATH; install Node.js/npm to install frontend dependencies.")
        return None

    print(f"Frontend dependencies are missing; running `npm install` in {frontend_dir}", flush=True)
    completed = subprocess.run([npm, "install"], cwd=frontend_dir)
    if completed.returncode != 0:
        _error(f"frontend dependency installation failed; fix the npm error above and rerun `{prog}`.")
        return None

    next_bin = _find_next_binary(frontend_dir)
    if next_bin is None:
        _error("frontend dependency installation completed, but the local Next.js executable is still missing.")
    return next_bin


def _find_next_binary(frontend_dir: Path) -> Path | None:
    """Path to the locally installed Next.js executable, or ``None`` if absent.

    On Windows the runnable shim is ``next.cmd``; elsewhere it is ``next``.
    """
    name = "next.cmd" if os.name == "nt" else "next"
    candidate = frontend_dir / "node_modules" / ".bin" / name
    return candidate if candidate.exists() else None


def _run_studio(
    backend: tuple[list[str], Path, dict[str, str], str],
    frontend: tuple[list[str], Path, dict[str, str], str],
) -> int:
    """Start both processes, then wait until one exits or Ctrl+C is received."""
    processes: list[subprocess.Popen[bytes]] = []
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        _terminate(processes)

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        for cmd, cwd, env, message in (backend, frontend):
            if stopping:
                return 0
            print(message, flush=True)
            processes.append(subprocess.Popen(cmd, cwd=cwd, env=env))
        return _wait_for_exit(processes, lambda: stopping)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        _terminate(processes)


def _wait_for_exit(
    processes: list[subprocess.Popen[bytes]],
    is_stopping: Callable[[], bool],
) -> int:
    """Wait until one managed subprocess exits, then return its exit code."""
    while True:
        for process in processes:
            code = process.poll()
            if code is None:
                continue
            _terminate([p for p in processes if p is not process])
            return 0 if is_stopping() else code
        time.sleep(0.2)


def _terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    """Terminate managed subprocesses, escalating to kill after 10s."""
    live = [process for process in processes if process.poll() is None]
    for process in live:
        process.terminate()

    deadline = time.monotonic() + 10
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


def _error(message: str) -> int:
    """Print an error to stderr and return the launcher's failure exit code."""
    print(message, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
