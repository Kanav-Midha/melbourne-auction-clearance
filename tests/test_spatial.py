"""Spatial feature tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.spatial import haversine_km, station_proximity


def test_haversine_zero_distance():
    assert haversine_km(-37.8183, 144.9671, -37.8183, 144.9671) == pytest.approx(0.0)


def test_haversine_known_distance():
    """Flinders Street to Box Hill station is about 13.5 km as the crow flies."""
    d = haversine_km(-37.8183, 144.9671, -37.8190, 145.1220)
    assert 13.0 < d < 14.0


def test_haversine_is_symmetric():
    a = haversine_km(-37.80, 144.96, -37.95, 145.15)
    b = haversine_km(-37.95, 145.15, -37.80, 144.96)
    assert a == pytest.approx(b)


def test_haversine_broadcasts():
    lats = np.array([-37.80, -37.90, -38.00])
    out = haversine_km(lats, 144.96, -37.8183, 144.9671)
    assert out.shape == (3,)
    assert np.all(np.diff(out) > 0)  # monotonically further from the CBD


def test_haversine_matches_one_degree_of_latitude():
    """One degree of latitude is ~111.2 km anywhere on the globe."""
    assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.1)


def test_station_proximity_picks_the_nearest():
    props = pd.DataFrame({"lat": [-37.80], "lon": [144.96]})
    stations = pd.DataFrame({
        "stop_lat": [-37.801, -37.90, -38.10],
        "stop_lon": [144.961, 145.00, 145.30],
    })
    out = station_proximity(props, stations)
    assert out["distance_to_station_km"].iloc[0] < 0.2
    assert out["stations_within_1km"].iloc[0] == 1


def test_station_proximity_counts_within_radius():
    props = pd.DataFrame({"lat": [-37.80], "lon": [144.96]})
    # Three stations inside 1 km, one far away.
    stations = pd.DataFrame({
        "stop_lat": [-37.801, -37.803, -37.806, -38.50],
        "stop_lon": [144.961, 144.962, 144.958, 145.90],
    })
    out = station_proximity(props, stations)
    assert out["stations_within_1km"].iloc[0] == 3


def test_station_proximity_is_chunk_invariant():
    """Results must not depend on how the distance matrix is chunked."""
    rng = np.random.default_rng(0)
    props = pd.DataFrame({
        "lat": -37.8 + rng.normal(0, 0.1, 120),
        "lon": 144.96 + rng.normal(0, 0.1, 120),
    })
    stations = pd.DataFrame({
        "stop_lat": -37.8 + rng.normal(0, 0.1, 25),
        "stop_lon": 144.96 + rng.normal(0, 0.1, 25),
    })
    full = station_proximity(props, stations)
    halves = pd.concat([
        station_proximity(props.iloc[:50], stations),
        station_proximity(props.iloc[50:], stations),
    ])
    pd.testing.assert_frame_equal(full, halves)
