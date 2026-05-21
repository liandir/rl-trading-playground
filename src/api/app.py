"""FastAPI application factory for the studio backend."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import agents, checkpoints, data, envs, files, meta, runs, ws
from src.api.services.events import get_broker
from src.api.services.runner import get_runner
from src.api.services.store import get_store
from src.api.settings import get_settings


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    runner = get_runner()
    await runner.start()
    try:
        yield
    finally:
        await runner.stop()
        await get_broker().shutdown()


def create_app() -> FastAPI:
    """Build the FastAPI app. Side effect: ensures the on-disk store exists."""

    settings = get_settings()
    settings.ensure_dirs()
    get_store()  # initialise the SQLite schema eagerly

    app = FastAPI(
        title="Trading Studio API",
        version="0.1.0",
        description="Backend for the trading studio: agents, environments, training runs.",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins) or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(meta.router)
    app.include_router(agents.router)
    app.include_router(envs.router)
    app.include_router(data.router)
    app.include_router(runs.router)
    app.include_router(checkpoints.router)
    app.include_router(files.router)
    app.include_router(ws.router)
    return app


app = create_app()
