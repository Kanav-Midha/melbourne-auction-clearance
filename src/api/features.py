"""Turn an API request into the exact feature row the model was trained on.

This module is the online half of the feature pipeline, and it is the part most
likely to quietly break. The offline pipeline builds features for 46,000 rows
with full history; this one builds a single row from a property plus a context
snapshot. If the two disagree about a column name, a dtype, or a category
level, the model still returns a number -- it just returns a wrong one.

The defences:

* the feature row is built as a DataFrame with ``config.FEATURES`` column order
  and then reindexed to the model's own ``feature_name()``;
* categorical levels come from the saved training bundle, not from the request;
* anything the context cannot supply is left as NaN, which LightGBM handles
  natively, rather than being silently defaulted to zero.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.api.context import MarketContext
from src.api.schemas import PropertyRequest
from src.features.preprocess import _SEASONS, _VIC_SCHOOL_HOLIDAYS
from src.features.price_model import VALUE_ATTRS, VALUE_CATEGORICALS

log = logging.getLogger(__name__)


def _school_holiday(d: pd.Timestamp) -> int:
    mmdd = d.strftime("%m-%d")
    for start, end in _VIC_SCHOOL_HOLIDAYS:
        if start <= end:
            if start <= mmdd <= end:
                return 1
        elif mmdd >= start or mmdd <= end:
            return 1
    return 0


def build_feature_row(req: PropertyRequest, ctx: MarketContext,
                      bundle: dict) -> tuple[pd.DataFrame, dict, list[str]]:
    """Return ``(feature_frame, drivers, warnings)`` for one property."""
    warnings: list[str] = []
    date = pd.Timestamp(req.auction_date)

    try:
        sub = ctx.lookup(req.suburb)
    except KeyError:
        raise KeyError(req.suburb) from None

    if date < ctx.as_of:
        warnings.append(
            f"auction_date {req.auction_date} precedes the context snapshot "
            f"({ctx.as_of.date()}); the market features describe the snapshot, not that date"
        )
    elif (date - ctx.as_of).days > 120:
        warnings.append(
            f"auction_date is {(date - ctx.as_of).days} days beyond the context "
            f"snapshot ({ctx.as_of.date()}); trailing market features are stale"
        )

    rain, tmax = ctx.weather_for(date)
    if req.rainfall_mm is not None:
        rain = req.rainfall_mm
    if req.max_temp_c is not None:
        tmax = req.max_temp_c

    guide_mid = req.guide_midpoint
    suburb_median = float(sub["suburb_median_price_l90d"]) if pd.notna(
        sub["suburb_median_price_l90d"]) else np.nan

    year_built = req.year_built if req.year_built is not None else np.nan
    age = float(date.year - year_built) if not pd.isna(year_built) else np.nan
    if not pd.isna(age):
        age = max(age, 0.0)

    row = {
        "bedrooms": float(req.bedrooms),
        "bathrooms": float(req.bathrooms),
        "car_spaces": float(req.car_spaces),
        "land_size_sqm": float(req.land_size_sqm) if req.land_size_sqm is not None else np.nan,
        "building_area_sqm": float(req.building_area_sqm) if req.building_area_sqm is not None else np.nan,
        "property_age_years": age,
        "distance_to_cbd_km": float(sub["distance_to_cbd_km"]),
        "distance_to_station_km": float(sub["distance_to_station_km"]),
        "stations_within_1km": float(sub["stations_within_1km"]),
        "guide_price_midpoint": guide_mid,
        "guide_vs_suburb_median": guide_mid / suburb_median if suburb_median else np.nan,
        "suburb_median_price_l90d": suburb_median,
        "suburb_clearance_l4w": float(sub["suburb_clearance_l4w"])
            if pd.notna(sub["suburb_clearance_l4w"]) else np.nan,
        "region_clearance_l4w": float(
            ctx.regions.loc[sub["region_name"], "region_clearance_l4w"]
        ) if sub["region_name"] in ctx.regions.index else np.nan,
        "auctions_scheduled_that_day": ctx.volume_by_month.get(int(date.month), np.nan),
        "auctions_in_suburb_that_day": np.nan,
        "rba_cash_rate": ctx.macro["rba_cash_rate"],
        "rba_cash_rate_change_3m": ctx.macro["rba_cash_rate_change_3m"],
        "rainfall_mm": rain,
        "max_temp_c": tmax,
        "week_of_year": float(date.isocalendar().week),
        "month": float(date.month),
        "school_holiday_flag": float(_school_holiday(date)),
        "property_type": req.property_type,
        "suburb": req.suburb,
        "council_area": str(sub["council_area"]),
        "region_name": str(sub["region_name"]),
        "agency": req.agency,
        "season": _SEASONS[date.month],
    }

    # --- expected value + overpricing gap ------------------------------------
    value_booster = bundle.get("value_booster")
    expected_value = np.nan
    if value_booster is not None:
        vlevels = bundle["value_levels"]
        vframe = pd.DataFrame([{k: row.get(k) for k in VALUE_ATTRS}])
        for c in VALUE_ATTRS:
            if c in VALUE_CATEGORICALS:
                vframe[c] = pd.Categorical(vframe[c].astype(str), categories=vlevels[c])
            else:
                vframe[c] = pd.to_numeric(vframe[c], errors="coerce").astype("float32")
        row["expected_log_value"] = float(value_booster.predict(vframe)[0])
        expected_value = float(np.exp(row["expected_log_value"]))
        row["price_expectation_gap"] = float(np.log(guide_mid) - row["expected_log_value"])
    else:
        row["expected_log_value"] = np.nan
        row["price_expectation_gap"] = np.nan
        warnings.append("value model unavailable; overpricing features are null")

    # --- assemble in the model's own column order ----------------------------
    X = pd.DataFrame([row])
    for c in config.NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    for c in config.CATEGORICAL_FEATURES:
        levels = bundle["category_levels"][c]
        value = X[c].iloc[0]
        if value is not None and str(value) not in set(map(str, levels)):
            warnings.append(f"unseen {c} '{value}'; treated as unknown")
            X[c] = pd.Categorical([None], categories=levels)
        else:
            X[c] = pd.Categorical(
                [None if value is None else str(value)], categories=list(map(str, levels))
            )

    X = X[bundle["features"]]

    drivers = {
        "suburb_clearance_l4w": _opt(row["suburb_clearance_l4w"]),
        "region_clearance_l4w": _opt(row["region_clearance_l4w"]),
        "guide_vs_expected_value_pct": _opt(
            100.0 * (np.exp(row["price_expectation_gap"]) - 1.0)
            if pd.notna(row["price_expectation_gap"]) else np.nan
        ),
        "expected_value": _opt(round(expected_value, -3) if pd.notna(expected_value) else np.nan),
        "rba_cash_rate": _opt(row["rba_cash_rate"]),
        "auctions_scheduled_that_day": _opt(row["auctions_scheduled_that_day"]),
    }
    return X, drivers, warnings


def _opt(v) -> float | None:
    """NaN -> None, so it serialises as JSON null rather than a NaN literal."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return None
    return round(float(v), 4)


__all__ = ["build_feature_row"]
