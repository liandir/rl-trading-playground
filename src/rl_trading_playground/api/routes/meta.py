"""Health, version, and registry endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from rl_trading_playground.api.schemas.registry import RegistryResponse
from rl_trading_playground.api.services import registry


router = APIRouter(tags=["meta"])


class HealthResponse(BaseModel):
    status: str
    time: datetime


class VersionResponse(BaseModel):
    name: str
    version: str
    api: str


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok", time=datetime.now(timezone.utc))


@router.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(name="trading-studio", version="0.1.0", api="v1")


@router.get("/registry", response_model=RegistryResponse)
def get_registry() -> RegistryResponse:
    return registry.response()
