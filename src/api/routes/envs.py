"""CRUD endpoints for saved environment presets."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.schemas.env import EnvironmentConfig, EnvironmentRecord
from src.api.services.store import get_store


router = APIRouter(prefix="/envs", tags=["envs"])


class CreateEnvironmentRequest(BaseModel):
    name: str = Field(..., min_length=1)
    config: EnvironmentConfig


@router.get("", response_model=list[EnvironmentRecord])
def list_envs() -> list[EnvironmentRecord]:
    return get_store().list_envs()


@router.post("", response_model=EnvironmentRecord, status_code=201)
def create_env(body: CreateEnvironmentRequest) -> EnvironmentRecord:
    return get_store().create_env(body.name, body.config)


@router.get("/{env_id}", response_model=EnvironmentRecord)
def get_env(env_id: str) -> EnvironmentRecord:
    rec = get_store().get_env(env_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="environment not found")
    return rec


@router.delete("/{env_id}", status_code=204)
def delete_env(env_id: str) -> None:
    if not get_store().delete_env(env_id):
        raise HTTPException(status_code=404, detail="environment not found")
