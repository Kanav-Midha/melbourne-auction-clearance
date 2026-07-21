"""Spatial feature engineering.

Melbourne buyers price transport access, so proximity to the PTV metropolitan
train network and distance from the CBD are the two spatial signals that carry
most of the weight. Everything here is vectorised -- the original notebook
looped over 48k rows with ``.apply`` and took four minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Broadcasts over array arguments.

    >>> float(haversine_km(-37.8183, 144.9671, -37.8183, 144.9671))
    0.0
    >>> d = haversine_km(-37.8183, 144.9671, -37.7986, 144.9784)  # Flinders St -> Fitzroy
    >>> bool(2.0 < d < 3.0)
    True
    """
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float))
                              for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def distance_to_cbd(df: pd.DataFrame) -> pd.Series:
    """Kilometres from Flinders Street Station."""
    return pd.Series(
        haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy(),
                     config.CBD_LAT, config.CBD_LON),
        index=df.index,
        name="distance_to_cbd_km",
    )


def station_proximity(df: pd.DataFrame, stations: pd.DataFrame,
                      radius_km: float = 1.0) -> pd.DataFrame:
    """Distance to the nearest train station and station count within `radius_km`.

    Uses a chunked distance matrix: 48k properties x ~170 stations is small
    enough to brute-force, and a BallTree adds a dependency for no real gain at
    this scale. Chunking keeps peak memory flat if the property set grows.
    """
    s_lat = stations["stop_lat"].to_numpy(dtype=float)
    s_lon = stations["stop_lon"].to_numpy(dtype=float)
    p_lat = df["lat"].to_numpy(dtype=float)
    p_lon = df["lon"].to_numpy(dtype=float)

    nearest = np.empty(len(df), dtype=float)
    within = np.empty(len(df), dtype=int)

    chunk = 5000
    for start in range(0, len(df), chunk):
        stop = min(start + chunk, len(df))
        d = haversine_km(
            p_lat[start:stop, None], p_lon[start:stop, None], s_lat[None, :], s_lon[None, :]
        )
        nearest[start:stop] = d.min(axis=1)
        within[start:stop] = (d <= radius_km).sum(axis=1)

    return pd.DataFrame(
        {"distance_to_station_km": nearest, "stations_within_1km": within},
        index=df.index,
    )


def add_spatial_features(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Attach all spatial features in one pass."""
    out = df.copy()
    out["distance_to_cbd_km"] = distance_to_cbd(out)
    out = out.join(station_proximity(out, stations))
    return out


__all__ = [
    "haversine_km",
    "distance_to_cbd",
    "station_proximity",
    "add_spatial_features",
]
