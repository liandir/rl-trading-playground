"""Market data loading configuration."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


DataSource = Literal["auto", "prepared", "kraken_csv", "synthetic"]


class DataConfig(BaseModel):
    """Options for loading historical market data into the runner."""

    source: DataSource = Field("auto", title="Source", description="auto picks prepared if present, else CSV.")
    prepared_path: str = Field(
        "data/historical_data0.ptt",
        title="Prepared file",
        description="Path to a torch.save'd dict with open/high/low/close/volume/times/pairs.",
    )
    raw_base_path: str = Field(
        "data/Kraken_OHLCVT",
        title="Raw CSV folder",
        description="Folder containing the raw Kraken OHLCVT CSV exports.",
    )
    interval: int = Field(5, ge=1, le=1440, title="CSV interval (min)")
    train_split: float = Field(0.9, ge=0.05, le=0.99, title="Train fraction")


class DataSourceRecord(BaseModel):
    """A persisted, named data source preset."""

    id: str
    name: str
    config: DataConfig
    created_at: datetime
