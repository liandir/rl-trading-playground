"""Runtime settings and on-disk paths for the studio backend."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Resolved paths and feature flags for the running API instance."""

    store_root: Path
    db_path: Path
    runs_dir: Path
    agents_dir: Path
    envs_dir: Path
    data_sources_dir: Path
    deployments_dir: Path
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        store_root = Path(os.environ.get("STUDIO_STORE", REPO_ROOT / "data" / "store")).resolve()
        origins_env = os.environ.get("STUDIO_CORS", "http://localhost:3000,http://127.0.0.1:3000")
        origins = tuple(o.strip() for o in origins_env.split(",") if o.strip())
        return cls(
            store_root=store_root,
            db_path=store_root / "studio.db",
            runs_dir=store_root / "runs",
            agents_dir=store_root / "agents",
            envs_dir=store_root / "envs",
            data_sources_dir=store_root / "data_sources",
            deployments_dir=store_root / "deployments",
            cors_origins=origins,
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.store_root,
            self.runs_dir,
            self.agents_dir,
            self.envs_dir,
            self.data_sources_dir,
            self.deployments_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
        _settings.ensure_dirs()
    return _settings
