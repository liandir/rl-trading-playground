"""Environment configuration and persisted env records."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


EnvironmentType = Literal["longshort_hierarchical_leverage"]


class EnvironmentConfig(BaseModel):
    """Construction options for the trading environment."""

    env_type: EnvironmentType = Field("longshort_hierarchical_leverage", title="Type")
    cash: float = Field(1_000.0, gt=0.0, title="Initial cash")
    tau_minutes: tuple[int, ...] = Field((30, 240, 1440, 10080), title="Tau windows (min)")
    bankruptcy_threshold: float = Field(10.0, ge=0.0, title="Bankruptcy threshold")
    min_open_dollars: float = Field(2.0, ge=0.0, title="Min open ($)")
    transaction_eps: float = Field(1e-2, ge=0.0, title="Transaction epsilon")
    use_dollar_volume: bool = True
    size_buckets: tuple[float, ...] = Field((0.10, 0.25, 0.50, 0.75, 0.90, 1.00), title="Size buckets")
    close_fee: float = Field(1.0, ge=0.0, title="Close fee (bps)")
    open_fee: float = Field(1.0, ge=0.0, title="Open fee (bps)")
    tax_rate: float = Field(0.26, ge=0.0, le=1.0, title="Tax rate")
    reward_mode: str = Field("log", title="Reward mode")
    val_coeff: float = Field(20.0, title="Value coefficient")
    roi_coeff: float = Field(100.0, title="ROI coefficient")
    done_reward_penalty: float = Field(100.0, title="Done penalty")
    max_leverage: float = Field(10.0, ge=1.0, title="Max leverage")
    maintenance_margin_ratio: float | None = Field(None, title="Maintenance margin")
    dtype: Literal["float32", "float64"] = Field("float32", title="Tensor dtype")
    device: str = Field("cpu", title="Torch device")
    eps: float = Field(1e-8, gt=0.0, title="Numeric epsilon")


class EnvironmentRecord(BaseModel):
    """A saved environment preset."""

    id: str
    name: str
    config: EnvironmentConfig
    created_at: datetime
