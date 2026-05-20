"""Test fixtures for the studio API."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api import settings as settings_module
from src.api.services import events as events_module
from src.api.services import runner as runner_module
from src.api.services import store as store_module
from src.api.settings import Settings


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    cfg = Settings(
        store_root=tmp_path / "store",
        db_path=tmp_path / "store" / "studio.db",
        runs_dir=tmp_path / "store" / "runs",
        agents_dir=tmp_path / "store" / "agents",
        envs_dir=tmp_path / "store" / "envs",
        data_sources_dir=tmp_path / "store" / "data_sources",
        cors_origins=("http://localhost:3000",),
    )
    monkeypatch.setattr(settings_module, "_settings", cfg)
    monkeypatch.setenv("STUDIO_STORE", str(cfg.store_root))
    return cfg


@pytest.fixture()
def store(settings: Settings) -> store_module.Store:
    runner_module.reset_runner_for_tests()
    events_module.reset_broker_for_tests()
    return store_module.reset_store_for_tests(settings)


@pytest.fixture()
def client(store: store_module.Store) -> Iterator[TestClient]:
    app = app_module.create_app()
    with TestClient(app) as client:
        yield client
