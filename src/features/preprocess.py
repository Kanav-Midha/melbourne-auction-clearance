"""Feature construction.

The single most important rule in this module: **every aggregate is computed
from auctions that happened strictly before the auction being scored.**

The first version of the trailing clearance rate was a one-character mistake:

    g[["sold", "held"]].rolling("28D").sum()            # includes today
    g[["sold", "held"]].rolling("28D", closed="left").sum()   # correct

The first form tells every row how its own auction day went. It lifts AUC from
0.644 to 0.775 and is worth exactly nothing in production, because at 9am on
Saturday you do not know how Saturday went. ``closed="left"`` is what makes the
window causal, and ``leakage_audit.check_temporal_integrity`` is what stops it
regressing.

Features that *are* legitimately known before the hammer falls:
  * the published auction calendar (volumes are known by Thursday)
  * the vendor's price guide
  * the RBA cash rate
  * the BOM forecast (we use the observation as a proxy for the forecast)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

CLEARANCE_WINDOW = "28D"   # four auction weekends
PRICE_WINDOW = "90D"       # a quarter of sales, enough for a stable median

_SEASONS = {
    12: "summer", 1: "summer", 2: "summer",
    3: "autumn", 4: "autumn", 5: "autumn",
    6: "winter", 7: "winter", 8: "winter",
    9: "spring", 10: "spring", 11: "spring",
}

# Victorian school holiday windows (approximate; DET term dates 2019-2024).
# Auction volumes and buyer attendance both drop inside these windows.
_VIC_SCHOOL_HOLIDAYS = [
    ("12-20", "01-28"),  # summer break, wraps the year boundary
    ("04-05", "04-22"),  # term 1 break (Easter)
    ("06-25", "07-14"),  # term 2 break
    ("09-17", "10-04"),  # term 3 break
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Week, month, season and a Victorian school-holiday flag."""
    out = df.copy()
    d = out["auction_date"]
    out["week_of_year"] = d.dt.isocalendar().week.astype("int16")
    out["month"] = d.dt.month.astype("int8")
    out["season"] = out["month"].map(_SEASONS).astype("string")

    mmdd = d.dt.strftime("%m-%d")
    flag = pd.Series(False, index=out.index)
    for start, end in _VIC_SCHOOL_HOLIDAYS:
        if start <= end:
            flag |= (mmdd >= start) & (mmdd <= end)
        else:  # window wraps 31 December
            flag |= (mmdd >= start) | (mmdd <= end)
    out["school_holiday_flag"] = flag.astype("int8")
    return out


def add_supply_features(df: pd.DataFrame) -> pd.DataFrame:
    """Auction volume on the day, city-wide and in the suburb.

    Not leakage: the auction calendar is published days in advance, so the
    number of competing auctions is known before the campaign closes.
    """
    out = df.copy()
    out["auctions_scheduled_that_day"] = (
        out.groupby("auction_date")["listing_id"].transform("size").astype("int32")
    )
    out["auctions_in_suburb_that_day"] = (
        out.groupby(["auction_date", "suburb"])["listing_id"].transform("size").astype("int16")
    )
    return out


def _trailing_rate(df: pd.DataFrame, key: str, window: str) -> pd.Series:
    """Clearance rate over `window`, excluding the current day.

    Aggregates to (key, date) first so the rolling window sees one row per
    auction day; every auction on a given Saturday therefore shares the same
    prior history, which is exactly the information a user would have.
    """
    daily = (
        df.groupby([key, "auction_date"])[config.TARGET]
        .agg(sold="sum", held="size")
        .reset_index()
        .sort_values([key, "auction_date"])
    )

    pieces = []
    for name, grp in daily.groupby(key, sort=False):
        g = grp.set_index("auction_date")
        roll = g[["sold", "held"]].rolling(window, closed="left").sum()
        rate = (roll["sold"] / roll["held"]).rename("rate")
        pieces.append(rate.reset_index().assign(**{key: name}))

    trailing = pd.concat(pieces, ignore_index=True)
    merged = df[[key, "auction_date"]].merge(trailing, on=[key, "auction_date"], how="left")
    return pd.Series(merged["rate"].to_numpy(), index=df.index)


def _trailing_median_price(df: pd.DataFrame, window: str) -> pd.Series:
    """Median sold price in the suburb over `window`, excluding the current day.

    Uses ``sold_price``, which is a leaky column at the row level -- but the
    *historical* sold prices of *other* properties are public record and known
    before the auction. The ``closed="left"`` window is what makes this safe;
    see ``src/features/leakage_audit.py`` for the test that enforces it.
    """
    daily = (
        df.groupby(["suburb", "auction_date"])["sold_price"]
        .median()
        .reset_index()
        .sort_values(["suburb", "auction_date"])
    )

    pieces = []
    for name, grp in daily.groupby("suburb", sort=False):
        g = grp.set_index("auction_date")
        med = g["sold_price"].rolling(window, closed="left").median().rename("med")
        pieces.append(med.reset_index().assign(suburb=name))

    trailing = pd.concat(pieces, ignore_index=True)
    merged = df[["suburb", "auction_date"]].merge(trailing, on=["suburb", "auction_date"], how="left")
    return pd.Series(merged["med"].to_numpy(), index=df.index)


def add_trailing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Suburb and region momentum, plus the trailing suburb median price."""
    out = df.copy()
    out["suburb_clearance_l4w"] = _trailing_rate(out, "suburb", CLEARANCE_WINDOW)
    out["region_clearance_l4w"] = _trailing_rate(out, "region_name", CLEARANCE_WINDOW)
    out["suburb_median_price_l90d"] = _trailing_median_price(out, PRICE_WINDOW)

    coverage = out[["suburb_clearance_l4w", "suburb_median_price_l90d"]].notna().mean()
    log.info("trailing feature coverage: %s",
             {k: f"{v:.1%}" for k, v in coverage.items()})
    return out


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cash rate level and the three-month change that buyers actually react to."""
    out = df.copy()
    monthly = (
        out.groupby(out["auction_date"].dt.to_period("M"))["rba_cash_rate"]
        .first()
        .sort_index()
    )
    change = (monthly - monthly.shift(3)).rename("rba_cash_rate_change_3m")
    out["rba_cash_rate_change_3m"] = (
        out["auction_date"].dt.to_period("M").map(change).astype("float64")
    )
    return out


def add_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """Property age and how the vendor's guide compares to the suburb median."""
    out = df.copy()
    out["property_age_years"] = (
        out["auction_date"].dt.year - out["year_built"]
    ).clip(lower=0).astype("float64")

    # The single strongest behavioural signal: a guide well above the suburb's
    # recent medians is a vendor with an expectation problem.
    out["guide_vs_suburb_median"] = (
        out["guide_price_midpoint"] / out["suburb_median_price_l90d"]
    ).replace([np.inf, -np.inf], np.nan)
    return out


def build_features(auctions: pd.DataFrame, stations: pd.DataFrame,
                   weather: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline: raw clean frame -> model-ready frame."""
    from src.data.weather import attach_weather
    from src.features.spatial import add_spatial_features

    df = auctions.sort_values("auction_date").reset_index(drop=True)
    df = add_spatial_features(df, stations)
    df = attach_weather(df, weather)
    df = add_calendar_features(df)
    df = add_supply_features(df)
    df = add_macro_features(df)
    df = add_trailing_features(df)      # must precede add_property_features
    df = add_property_features(df)      # depends on suburb_median_price_l90d

    expected = [c for c in config.FEATURES if c not in config.SPLIT_AWARE_FEATURES]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"feature pipeline did not produce: {missing}")
    return df


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by auction date. Never shuffle a time series."""
    d = df["auction_date"]
    train = df[d <= config.TRAIN_END]
    valid = df[(d > config.TRAIN_END) & (d <= config.VALID_END)]
    test = df[d > config.VALID_END]

    log.info(
        "split -> train %d (%s..%s) | valid %d | test %d",
        len(train), d.min().date(), config.TRAIN_END, len(valid), len(test),
    )
    return train.copy(), valid.copy(), test.copy()


__all__ = [
    "add_calendar_features", "add_supply_features", "add_trailing_features",
    "add_macro_features", "add_property_features", "build_features",
    "chronological_split",
]
