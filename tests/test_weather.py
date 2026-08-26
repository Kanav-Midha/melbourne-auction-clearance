"""BOM weather parsing and gap-filling tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.weather import WEATHER_COLS, attach_weather, fill_weather_gaps
from tests.conftest import requires_data


def _raw(n_days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2021-06-01", periods=n_days, freq="D")
    frames = []
    for i, sid in enumerate(["086338", "086282", "086038"]):
        frames.append(pd.DataFrame({
            "station_id": sid,
            "station_name": f"STATION {i}",
            "station_lat": -37.8 - 0.1 * i,
            "station_lon": 144.9 + 0.1 * i,
            "date": dates,
            "max_temp_c": 15.0 + i + np.arange(n_days) * 0.05,
            "rainfall_mm": 2.0 + i,
        }))
    return pd.concat(frames, ignore_index=True)


class TestFillWeatherGaps:
    def test_leaves_complete_data_untouched(self):
        raw = _raw()
        out = fill_weather_gaps(raw)
        assert (out["weather_source"] == "observed").all()
        np.testing.assert_allclose(out["max_temp_c"], raw["max_temp_c"])

    def test_fills_a_single_station_outage_from_the_network(self):
        raw = _raw()
        outage = (raw["station_id"] == "086338") & (raw["date"] < "2021-06-10")
        raw.loc[outage, WEATHER_COLS] = np.nan

        out = fill_weather_gaps(raw)
        assert out[WEATHER_COLS].notna().all().all()
        filled = out[outage.to_numpy()]
        assert (filled["weather_source"] == "network_daily").all()
        # Filled value is the mean of the two reporting stations.
        assert filled["rainfall_mm"].iloc[0] == pytest.approx((3.0 + 4.0) / 2)

    def test_falls_back_to_climatology_when_nobody_reported(self):
        raw = _raw()
        blackout = raw["date"] == "2021-06-15"
        raw.loc[blackout, WEATHER_COLS] = np.nan

        out = fill_weather_gaps(raw)
        assert out[WEATHER_COLS].notna().all().all()
        assert (out.loc[blackout.to_numpy(), "weather_source"] == "climatology").all()

    def test_raises_rather_than_returning_nulls(self):
        raw = _raw()
        raw[WEATHER_COLS] = np.nan
        with pytest.raises(ValueError, match="unfilled"):
            fill_weather_gaps(raw)


@requires_data
class TestAttachWeather:
    def test_every_auction_gets_weather(self, auctions, weather):
        out = attach_weather(auctions, weather)
        assert out["rainfall_mm"].notna().all()
        assert out["max_temp_c"].notna().all()

    def test_nearest_station_is_actually_near(self, auctions, weather):
        """Melbourne metro is ~50km across; nothing should be further than that."""
        out = attach_weather(auctions, weather)
        assert out["station_distance_km"].max() < 50

    def test_the_2021_olympic_park_outage_does_not_drop_rows(self, auctions, weather):
        """The outage is why gap-filling runs before the join, not after."""
        out = attach_weather(auctions, weather)
        winter = out[(out["auction_date"] >= "2021-06-01") &
                     (out["auction_date"] <= "2021-08-31")]
        assert len(winter) > 0
        assert winter["rainfall_mm"].notna().all()
