"""Request and response models for the clearance API.

Validation is deliberately strict. The model was trained on Melbourne
residential auctions; a request for a 40-bedroom property in Sydney should be
rejected at the edge with a clear message rather than scored with a number
nobody can interpret.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PropertyType = Literal["h", "t", "u"]

PROPERTY_TYPE_LABELS = {
    "h": "House",
    "t": "Townhouse",
    "u": "Unit/Apartment",
}


class PropertyRequest(BaseModel):
    """A single property going to auction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "suburb": "Northcote",
                "auction_date": "2025-03-15",
                "property_type": "h",
                "bedrooms": 3,
                "bathrooms": 1,
                "car_spaces": 1,
                "land_size_sqm": 372.0,
                "building_area_sqm": 128.0,
                "year_built": 1925,
                "guide_price_low": 1480000,
                "guide_price_high": 1620000,
                "agency": "Nelson Alexander",
            }
        }
    )

    suburb: str = Field(..., min_length=2, max_length=60,
                        description="Melbourne metropolitan suburb name")
    auction_date: date = Field(..., description="Scheduled auction date")
    property_type: PropertyType = Field(..., description="h=house, t=townhouse, u=unit")

    bedrooms: int = Field(..., ge=0, le=12)
    bathrooms: int = Field(..., ge=0, le=8)
    car_spaces: int = Field(0, ge=0, le=10)

    land_size_sqm: float | None = Field(None, ge=0, le=20000)
    building_area_sqm: float | None = Field(None, ge=10, le=2000)
    year_built: int | None = Field(None, ge=1830, le=2035)

    guide_price_low: float = Field(..., gt=50_000, lt=100_000_000)
    guide_price_high: float = Field(..., gt=50_000, lt=100_000_000)

    agency: str | None = Field(None, max_length=80)

    # Optional overrides. Supplied by a caller that genuinely knows better than
    # the service's snapshot -- for example a BOM forecast for auction day.
    rainfall_mm: float | None = Field(None, ge=0, le=400)
    max_temp_c: float | None = Field(None, ge=-10, le=55)

    @field_validator("suburb")
    @classmethod
    def _tidy_suburb(cls, v: str) -> str:
        return v.strip().title()

    @field_validator("guide_price_high")
    @classmethod
    def _check_range(cls, v: float, info) -> float:
        low = info.data.get("guide_price_low")
        if low is not None and v < low:
            raise ValueError("guide_price_high must be >= guide_price_low")
        return v

    @property
    def guide_midpoint(self) -> float:
        return (self.guide_price_low + self.guide_price_high) / 2.0


class BatchRequest(BaseModel):
    """A campaign's worth of properties, scored in one call."""
    properties: list[PropertyRequest] = Field(..., min_length=1, max_length=500)


class Drivers(BaseModel):
    """The handful of context values that moved this prediction.

    Not SHAP values. These are the inputs an agent can actually act on or argue
    with, which is what makes the number trustworthy in a vendor conversation.
    """
    suburb_clearance_l4w: float | None = None
    region_clearance_l4w: float | None = None
    guide_vs_expected_value_pct: float | None = Field(
        None, description="Guide midpoint vs the model's expected value, in percent. "
                          "Positive means the guide is above the estimate."
    )
    expected_value: float | None = Field(
        None, description="Model estimate of the property's market value, AUD")
    rba_cash_rate: float | None = None
    auctions_scheduled_that_day: float | None = None


class PredictionResponse(BaseModel):
    """A calibrated probability plus the context it was computed from."""

    clearance_probability: float = Field(..., ge=0, le=1)
    predicted_outcome: Literal["likely_sell", "uncertain", "likely_pass_in"]
    confidence_band: str = Field(..., description="Coarse band for UI display")
    drivers: Drivers
    context_as_of: date
    model_version: str
    warnings: list[str] = Field(default_factory=list)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    summary: dict


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    context_loaded: bool
    context_as_of: date | None = None
    n_suburbs: int | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    trained_through: str
    train_rows: int
    n_features: int
    n_rounds: int
    test_metrics: dict
    top_features: list[dict]


__all__ = [
    "PropertyRequest", "BatchRequest", "PredictionResponse", "BatchResponse",
    "Drivers", "HealthResponse", "ModelInfoResponse", "PROPERTY_TYPE_LABELS",
]
