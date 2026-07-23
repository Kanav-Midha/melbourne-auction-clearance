"""BOM daily weather: parsing, gap-filling, and the nearest-station join.

The rain hypothesis is one of the few pieces of auction folklore that is easy to
test: a wet Saturday is supposed to thin the crowd on the nature strip and push
properties to pass in. Testing it needs a weather value for every auction, and
the BOM archive does not provide one.

Two distinct gaps exist in the raw extract:

1. **Scattered missing observations** -- roughly 7% of rainfall readings and 4%
   of temperature readings, spread uniformly. Cheap to fill from a neighbouring
   station on the same day.
2. **A three-month outage** at Melbourne (Olympic Park) over the 2021 winter.
   This one matters: Olympic Park is the nearest station to most inner-suburb
   auctions, so a naive nearest-station join silently drops ~9% of the 2021
   rows, and those rows are not missing at random -- they are concentrated in
   the inner suburbs, which clear well above average.

The fill is a three-step fallback, applied in order, with the step recorded in
``weather_source`` so the effect of imputation can be audited later.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.features.spatial import haversine_km

log = logging.getLogger(__name__)

WEATHER_COLS = ["rainfall_mm", "max_temp_c"]


def load_weather(path=None) -> pd.DataFrame:
    """Read the BOM extract and coerce it to numeric daily observations."""
    df = pd.read_csv(path or config.WEATHER_RAW, dtype={"rainfall_mm": "string"})
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

    # Rainfall ships as text, occasionally with a trailing quality flag.
    df["rainfall_mm"] = pd.to_numeric(
        df["rainfall_mm"].str.strip().replace({"": None}), errors="coerce"
    ).astype("float64")
    df["max_temp_c"] = pd.to_numeric(df["max_temp_c"], errors="coerce")

    missing = {c: float(df[c].isna().mean()) for c in WEATHER_COLS}
    log.info("loaded %d BOM observations; missing %s", len(df),
             {k: f"{v:.1%}" for k, v in missing.items()})
    return df.sort_values(["station_id", "date"]).reset_index(drop=True)


def _climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-year climatology per station, smoothed over a +/-7 day window.

    This is the last-resort fill. It is deliberately station-specific so that a
    bayside station does not inherit an inland station's summer maxima.
    """
    out = df.copy()
    out["doy"] = out["date"].dt.dayofyear
    clim = (
        out.groupby(["station_id", "doy"])[WEATHER_COLS]
        .mean()
        .reset_index()
        .sort_values(["station_id", "doy"])
    )
    for col in WEATHER_COLS:
        clim[col] = (
            clim.groupby("station_id")[col]
            .transform(lambda s: s.rolling(15, center=True, min_periods=1).mean())
        )
    return clim.rename(columns={c: f"{c}_clim" for c in WEATHER_COLS})


def fill_weather_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing observations, recording which rule supplied each value.

    Fallback order per station-day:
      1. ``observed``      -- the station reported.
      2. ``network_daily`` -- mean of the other stations that reported that day.
      3. ``climatology``   -- the station's own smoothed day-of-year normal.
    """
    out = df.copy()
    out["weather_source"] = np.where(
        out[WEATHER_COLS].isna().any(axis=1), "missing", "observed"
    )

    # --- step 2: same-day network mean ---------------------------------------
    network = out.groupby("date")[WEATHER_COLS].transform("mean")
    filled_by_network = pd.Series(False, index=out.index)
    for col in WEATHER_COLS:
        gap = out[col].isna() & network[col].notna()
        out.loc[gap, col] = network.loc[gap, col]
        filled_by_network |= gap

    # --- step 3: station climatology -----------------------------------------
    clim = _climatology(df)
    out["doy"] = out["date"].dt.dayofyear
    out = out.merge(clim, on=["station_id", "doy"], how="left")
    filled_by_clim = pd.Series(False, index=out.index)
    for col in WEATHER_COLS:
        gap = out[col].isna()
        out.loc[gap, col] = out.loc[gap, f"{col}_clim"]
        filled_by_clim |= gap

    out["weather_source"] = np.where(
        out["weather_source"] == "observed", "observed",
        np.where(filled_by_clim, "climatology", "network_daily"),
    )
    out = out.drop(columns=["doy"] + [f"{c}_clim" for c in WEATHER_COLS])

    still_missing = int(out[WEATHER_COLS].isna().sum().sum())
    if still_missing:
        raise ValueError(f"{still_missing} weather values remain unfilled")

    log.info("weather fill: %s", out["weather_source"].value_counts().to_dict())
    return out


def attach_weather(auctions: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join each auction to its nearest reporting station on the auction date.

    ``weather`` must already be gap-filled, otherwise the join reintroduces
    nulls for exactly the inner-suburb rows the outage removed.
    """
    stations = (
        weather[["station_id", "station_lat", "station_lon"]]
        .drop_duplicates("station_id")
        .reset_index(drop=True)
    )

    d = haversine_km(
        auctions["lat"].to_numpy()[:, None], auctions["lon"].to_numpy()[:, None],
        stations["station_lat"].to_numpy()[None, :], stations["station_lon"].to_numpy()[None, :],
    )
    nearest_idx = d.argmin(axis=1)

    keyed = auctions[["auction_date"]].copy()
    keyed["station_id"] = stations["station_id"].to_numpy()[nearest_idx]
    keyed["station_distance_km"] = d[np.arange(len(auctions)), nearest_idx]

    obs = weather[["station_id", "date"] + WEATHER_COLS + ["weather_source"]]
    merged = keyed.merge(
        obs, how="left", left_on=["station_id", "auction_date"], right_on=["station_id", "date"]
    ).drop(columns=["date"])

    out = auctions.copy()
    for col in WEATHER_COLS + ["weather_source", "station_id", "station_distance_km"]:
        out[col] = merged[col].to_numpy()

    unmatched = int(out["rainfall_mm"].isna().sum())
    if unmatched:
        log.warning("%d auctions have no weather observation after the join", unmatched)
    return out


def build_weather_table(path=None) -> pd.DataFrame:
    """Load and gap-fill in one call."""
    return fill_weather_gaps(load_weather(path))


__all__ = [
    "WEATHER_COLS", "load_weather", "fill_weather_gaps",
    "attach_weather", "build_weather_table",
]
