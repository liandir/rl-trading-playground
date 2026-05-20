"""Data source presets and quick previews."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.schemas.data import DataConfig, DataSourceRecord
from src.api.services.store import get_store


router = APIRouter(prefix="/data", tags=["data"])


class CreateDataSourceRequest(BaseModel):
    name: str = Field(..., min_length=1)
    config: DataConfig


class PreviewResponse(BaseModel):
    n_steps: int
    n_assets: int
    asset_names: list[str]
    first_time: float | None = None
    last_time: float | None = None


@router.get("/sources", response_model=list[DataSourceRecord])
def list_sources() -> list[DataSourceRecord]:
    return get_store().list_data_sources()


@router.post("/sources", response_model=DataSourceRecord, status_code=201)
def create_source(body: CreateDataSourceRequest) -> DataSourceRecord:
    return get_store().create_data_source(body.name, body.config)


@router.post("/preview", response_model=PreviewResponse)
def preview(config: DataConfig) -> PreviewResponse:
    """Load the data source and return shape info (without sending tensors)."""

    try:
        # Lazy import to avoid loading torch in the meta routes.
        from src.api.runner.build import load_market_data

        data = load_market_data(config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    times = data.times
    return PreviewResponse(
        n_steps=data.n_steps,
        n_assets=data.n_assets,
        asset_names=list(data.pairs.keys()),
        first_time=float(times[0]) if data.n_steps else None,
        last_time=float(times[-1]) if data.n_steps else None,
    )
