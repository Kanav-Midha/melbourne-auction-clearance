"""Load and clean the raw auction extract.

Extracted from ``notebooks/01_initial_exploration.ipynb``. The notebook version
re-read the CSV in six different cells with slightly different cleaning each
time, which is how the duplicate-row bug survived as long as it did.

Every function here is pure: raw frame in, clean frame out, no file writes.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

#: REIV result codes that count as a sale for clearance-rate purposes.
#: REIV counts sold-before and sold-after in the published rate, so we do too.
SOLD_CODES = {"S", "SP", "SA"}
UNSOLD_CODES = {"PI", "VB"}
#: Withdrawn properties never went to auction, so they are not in the denominator.
EXCLUDED_CODES = {"W"}

_CURRENCY_RE = re.compile(r"[^0-9.\-]")


def parse_currency(series: pd.Series) -> pd.Series:
    """Turn ``"$1,250,000"`` / ``"1250000"`` / ``""`` into floats.

    The feed is inconsistent: roughly half the sold prices carry a dollar sign
    and thousands separators, the rest are bare integers, and unsold rows are
    empty strings rather than nulls.
    """
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "-": pd.NA, "N/A": pd.NA, "nan": pd.NA})
    s = s.str.replace(_CURRENCY_RE, "", regex=True)
    # Force float64: pandas infers Int64 when every value is a whole number,
    # which then propagates a nullable dtype into arithmetic downstream.
    return pd.to_numeric(s, errors="coerce").astype("float64")


def parse_price_guide(series: pd.Series) -> pd.DataFrame:
    """Split a free-text guide (``"$900,000 - $990,000"``) into low/high/mid.

    Single-value guides (``"$900,000"``) get the same low and high. Anything
    unparseable becomes NaN rather than silently defaulting to zero -- an early
    version defaulted to 0 and quietly told the model that every unparsed
    listing was free.
    """
    s = series.astype("string").str.strip()
    parts = s.str.split(r"\s*-\s*", n=1, regex=True, expand=True)
    if parts.shape[1] == 1:
        parts[1] = pd.NA

    low = parse_currency(parts[0])
    high = parse_currency(parts[1])
    high = high.fillna(low)
    low = low.fillna(high)

    # Guard against a transposed range in the source.
    lo = np.minimum(low, high)
    hi = np.maximum(low, high)

    return pd.DataFrame(
        {
            "guide_price_low": lo,
            "guide_price_high": hi,
            "guide_price_midpoint": (lo + hi) / 2.0,
        },
        index=series.index,
    )


def normalise_suburb(series: pd.Series) -> pd.Series:
    """Collapse ``" richmond"``, ``"RICHMOND"``, ``"Richmond"`` to one label.

    Before this, ``df.suburb.nunique()`` reported 297 suburbs in a dataset that
    covers 100 -- and the suburb-level aggregates were being computed across
    all three spellings.
    """
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
        # Title-casing breaks the handful of names that are not simple words.
        .str.replace(r"\bMc([a-z])", lambda m: "Mc" + m.group(1).upper(), regex=True)
        .str.replace("Obrien", "OBrien", regex=False)
    )


def derive_target(result_code: pd.Series) -> pd.Series:
    """Map REIV result codes to the binary target.

    Returns ``<NA>`` for withdrawn lots so they can be dropped explicitly rather
    than being silently counted as failures to sell.
    """
    return pd.Series(
        np.where(
            result_code.isin(SOLD_CODES), 1,
            np.where(result_code.isin(UNSOLD_CODES), 0, np.nan),
        ),
        index=result_code.index,
        dtype="Float64",
    ).astype("Int64")


def load_auctions(path=None, drop_withdrawn: bool = True) -> pd.DataFrame:
    """Read the raw auction CSV and return a clean, deduplicated frame."""
    path = path or config.AUCTIONS_RAW
    df = pd.read_csv(path, dtype={"sold_price": "string", "price_guide": "string"})
    n_raw = len(df)

    # --- deduplicate ---------------------------------------------------------
    # Results get double-entered when a campaign is re-published. The listing_id
    # is the source's primary key, so an exact repeat is safe to drop.
    df = df.drop_duplicates(subset=["listing_id"], keep="first").reset_index(drop=True)
    n_dupes = n_raw - len(df)

    # --- types ---------------------------------------------------------------
    df["auction_date"] = pd.to_datetime(df["auction_date"], format="%Y-%m-%d")
    df["suburb"] = normalise_suburb(df["suburb"])
    df["council_area"] = df["council_area"].astype("string").str.strip()
    df["region_name"] = df["region_name"].astype("string").str.strip()
    df["property_type"] = df["property_type"].astype("string").str.strip().str.lower()

    df["sold_price"] = parse_currency(df["sold_price"])
    df = df.join(parse_price_guide(df["price_guide"]))

    for col in ("price_per_sqm", "vendor_discount_pct", "days_to_settle"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["sale_settlement_date"] = pd.to_datetime(df["sale_settlement_date"], errors="coerce")

    # --- target --------------------------------------------------------------
    df[config.TARGET] = derive_target(df["result_code"])
    n_withdrawn = int(df[config.TARGET].isna().sum())
    if drop_withdrawn:
        df = df[df[config.TARGET].notna()].reset_index(drop=True)
    df[config.TARGET] = df[config.TARGET].astype("int8")

    # --- sanity --------------------------------------------------------------
    df = df[df["auction_date"].notna()]
    df = df.sort_values("auction_date").reset_index(drop=True)

    log.info(
        "loaded %d auctions (%d raw, -%d duplicates, -%d withdrawn); "
        "clearance %.1f%%; %d suburbs",
        len(df), n_raw, n_dupes, n_withdrawn,
        100 * df[config.TARGET].mean(), df["suburb"].nunique(),
    )
    return df


def load_stations(path=None) -> pd.DataFrame:
    """PTV metropolitan train station coordinates."""
    df = pd.read_csv(path or config.STATIONS_RAW)
    return df.dropna(subset=["stop_lat", "stop_lon"]).reset_index(drop=True)


__all__ = [
    "SOLD_CODES", "UNSOLD_CODES", "EXCLUDED_CODES",
    "parse_currency", "parse_price_guide", "normalise_suburb",
    "derive_target", "load_auctions", "load_stations",
]
