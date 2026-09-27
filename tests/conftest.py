"""Shared fixtures.

The full pipeline takes a few seconds, so anything derived from the real data is
session-scoped. Unit tests that only need a handful of rows build their own
frames inline instead -- a test that depends on 46,000 rows to check a string
parser is a test that will eventually be deleted for being slow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402


def _data_present() -> bool:
    return config.AUCTIONS_RAW.exists() and config.WEATHER_RAW.exists()


requires_data = pytest.mark.skipif(
    not _data_present(),
    reason="raw data absent; run `python -m src.data.make_dataset` first",
)
requires_model = pytest.mark.skipif(
    not (config.MODEL_PATH.exists() and (config.MODEL_DIR / "market_context.joblib").exists()),
    reason="trained model absent; run `make train` first",
)


@pytest.fixture(scope="session")
def auctions():
    from src.data.data_loader import load_auctions
    return load_auctions()


@pytest.fixture(scope="session")
def stations():
    from src.data.data_loader import load_stations
    return load_stations()


@pytest.fixture(scope="session")
def weather():
    from src.data.weather import build_weather_table
    return build_weather_table()


@pytest.fixture(scope="session")
def features(auctions, stations, weather):
    from src.features.preprocess import build_features
    return build_features(auctions, stations, weather)


@pytest.fixture(scope="session")
def api_client():
    from fastapi.testclient import TestClient

    from src.api.main import app
    with TestClient(app) as client:
        yield client
