"""Market-context store for online scoring.

The training features are not all attributes of a property. Several describe the
*market* on the day it goes to auction:

    suburb_clearance_l4w, region_clearance_l4w, suburb_median_price_l90d,
    auctions_scheduled_that_day, rba_cash_rate, rba_cash_rate_change_3m,
    rainfall_mm, max_temp_c

A caller asking "will 12 Smith St clear?" knows the property. It does not know
the trailing clearance rate in Northcote, and it should not have to. If the API
made the caller supply those, every consumer would compute them slightly
differently and the online features would drift away from the offline ones --
the classic training/serving skew failure.

So the service owns them. This module snapshots the most recent value of every
contextual feature from the historical data and persists it next to the model.
At request time the API looks up the suburb, fills the context, and only then
scores.

Refresh cadence in a real deployment would be weekly, the morning after results
are published. Here it is regenerated whenever the model is retrained.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import pandas as pd

from src import config

log = logging.getLogger(__name__)

CONTEXT_PATH = config.MODEL_DIR / "market_context.joblib"


@dataclass
class MarketContext:
    """Everything the API needs to turn a property into a feature row."""

    suburbs: pd.DataFrame          # indexed by suburb
    regions: pd.DataFrame          # indexed by region_name
    macro: dict                    # cash rate + 3m change
    volume_by_month: dict          # month -> typical auctions that day
    weather_climatology: pd.DataFrame  # day-of-year -> rainfall, max temp
    as_of: pd.Timestamp

    def known_suburbs(self) -> list[str]:
        return sorted(self.suburbs.index.tolist())

    def lookup(self, suburb: str) -> pd.Series:
        """Case-insensitive suburb lookup. Raises KeyError if unknown."""
        key = str(suburb).strip().title()
        if key not in self.suburbs.index:
            raise KeyError(key)
        return self.suburbs.loc[key]

    def weather_for(self, date: pd.Timestamp) -> tuple[float, float]:
        """Climatological rainfall and max temp for a calendar day.

        A production service would call the BOM forecast API for dates inside
        the seven-day window. Climatology is the honest fallback beyond that,
        and it is what the model was trained to expect when an observation was
        missing.
        """
        doy = int(pd.Timestamp(date).dayofyear)
        if doy in self.weather_climatology.index:
            row = self.weather_climatology.loc[doy]
            return float(row["rainfall_mm"]), float(row["max_temp_c"])
        return float(self.weather_climatology["rainfall_mm"].mean()), \
               float(self.weather_climatology["max_temp_c"].mean())


def build_context_store(df: pd.DataFrame) -> MarketContext:
    """Snapshot the latest contextual values from the feature frame."""
    df = df.sort_values("auction_date")
    as_of = df["auction_date"].max()

    latest_suburb = (
        df.groupby("suburb")
        .agg(
            lat=("lat", "median"),
            lon=("lon", "median"),
            council_area=("council_area", "last"),
            region_name=("region_name", "last"),
            suburb_clearance_l4w=("suburb_clearance_l4w", "last"),
            suburb_median_price_l90d=("suburb_median_price_l90d", "last"),
            distance_to_cbd_km=("distance_to_cbd_km", "median"),
            distance_to_station_km=("distance_to_station_km", "median"),
            stations_within_1km=("stations_within_1km", "median"),
            n_auctions=("listing_id", "size"),
        )
    )
    # A suburb with no auctions in the last four weeks has a null trailing rate;
    # fall back to its region so the caller still gets a sane prediction.
    latest_region = df.groupby("region_name").agg(
        region_clearance_l4w=("region_clearance_l4w", "last")
    )
    latest_suburb["suburb_clearance_l4w"] = latest_suburb["suburb_clearance_l4w"].fillna(
        latest_suburb["region_name"].map(latest_region["region_clearance_l4w"])
    )

    last_row = df.iloc[-1]
    macro = {
        "rba_cash_rate": float(last_row["rba_cash_rate"]),
        "rba_cash_rate_change_3m": float(
            0.0 if pd.isna(last_row["rba_cash_rate_change_3m"])
            else last_row["rba_cash_rate_change_3m"]
        ),
    }

    recent = df[df["auction_date"] >= as_of - pd.Timedelta("365D")]
    volume_by_month = (
        recent.groupby(recent["auction_date"].dt.month)["auctions_scheduled_that_day"]
        .median()
        .to_dict()
    )

    clim = (
        df.assign(doy=df["auction_date"].dt.dayofyear)
        .groupby("doy")[["rainfall_mm", "max_temp_c"]]
        .mean()
        .reindex(range(1, 367))
        .interpolate()
        .bfill()
        .ffill()
    )

    ctx = MarketContext(
        suburbs=latest_suburb,
        regions=latest_region,
        macro=macro,
        volume_by_month={int(k): float(v) for k, v in volume_by_month.items()},
        weather_climatology=clim,
        as_of=as_of,
    )
    log.info("built market context as of %s covering %d suburbs",
             as_of.date(), len(latest_suburb))
    return ctx


def save_context(ctx: MarketContext, path=None) -> None:
    joblib.dump(ctx, path or CONTEXT_PATH)
    log.info("saved %s", path or CONTEXT_PATH)


def load_context(path=None) -> MarketContext:
    return joblib.load(path or CONTEXT_PATH)


__all__ = [
    "MarketContext", "CONTEXT_PATH", "build_context_store",
    "save_context", "load_context",
]
