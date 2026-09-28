"""Central configuration. Everything that used to be a hardcoded string in the
notebook lives here.

Paths are resolved relative to the repo root so the code works the same whether
it is invoked from the CLI, a notebook, or the API container.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

AUCTIONS_RAW = RAW_DIR / "vic_auction_results.csv"
WEATHER_RAW = RAW_DIR / "bom_melbourne_daily.csv"
STATIONS_RAW = RAW_DIR / "ptv_train_stations.csv"

TRAIN_SET = PROCESSED_DIR / "train.parquet"
VALID_SET = PROCESSED_DIR / "valid.parquet"
TEST_SET = PROCESSED_DIR / "test.parquet"

MODEL_PATH = MODEL_DIR / "clearance_lgbm.joblib"
BASELINE_PATH = MODEL_DIR / "clearance_baseline.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

def _ensure_working_dirs() -> None:
    """Create the working directories, tolerating a filesystem we cannot write.

    Importing this module must never require write access. The serving container
    ships only ``models/`` and ``src/``, runs as a non-root user, and has no
    reason to own a ``data/`` directory -- an unconditional ``mkdir`` here raised
    PermissionError on import and the container died before uvicorn bound a port.
    The same applies to any read-only root filesystem, which is a normal way to
    harden a deployment.

    Nothing is silently swallowed: code that actually writes still fails at the
    point of writing, where the error names the path it wanted.
    """
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODEL_DIR, REPORTS_DIR, FIGURES_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue


_ensure_working_dirs()

# --- Geography ---------------------------------------------------------------
# Flinders Street Station, the conventional "0km" point for Melbourne distances.
CBD_LAT, CBD_LON = -37.8183, 144.9671

# --- Modelling ---------------------------------------------------------------
TARGET = "sold_at_auction"
RANDOM_SEED = 42

# Auctions are a weekly, seasonal market. Splitting randomly would let the model
# learn from the future, so every split in this project is chronological.
# Train on 2019-2022, validate on the 2023 calendar year, test on 2024.
# An earlier split validated on six months (~4.2k auctions), which was too
# noisy to tune against -- trial-to-trial PR-AUC moved more than the
# differences between hyperparameter sets.
TRAIN_END = "2022-12-31"
VALID_END = "2023-12-31"  # test is everything after this

# Columns that encode the outcome and must never reach the model.
# See src/features/leakage_audit.py for how this list was built.
LEAKY_COLUMNS = [
    "sold_price",
    "result_code",
    "result_description",
    "price_per_sqm",
    "sale_settlement_date",
    "days_to_settle",
    "vendor_discount_pct",
]

CATEGORICAL_FEATURES = [
    "property_type",
    "suburb",
    "council_area",
    "region_name",
    "agency",
    "season",
]

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "car_spaces",
    "land_size_sqm",
    "building_area_sqm",
    "property_age_years",
    "distance_to_cbd_km",
    "distance_to_station_km",
    "stations_within_1km",
    "guide_price_midpoint",
    "guide_vs_suburb_median",
    # Built by src/features/price_model.py -- see that module for why the
    # ratio above is not enough on its own.
    "expected_log_value",
    "price_expectation_gap",
    "suburb_median_price_l90d",
    "suburb_clearance_l4w",
    "region_clearance_l4w",
    "auctions_scheduled_that_day",
    "auctions_in_suburb_that_day",
    "rba_cash_rate",
    "rba_cash_rate_change_3m",
    "rainfall_mm",
    "max_temp_c",
    "week_of_year",
    "month",
    "school_holiday_flag",
]

#: Features that cannot be built row-wise because they depend on which rows are
#: training rows. ``build_features`` deliberately does not produce these; they
#: are attached after the chronological split by ``models.dataset.prepare_splits``.
SPLIT_AWARE_FEATURES = [
    "expected_log_value",
    "price_expectation_gap",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
