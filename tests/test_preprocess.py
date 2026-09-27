"""Feature pipeline tests."""
from __future__ import annotations

import pandas as pd

from src import config
from src.features.preprocess import (
    add_calendar_features,
    add_supply_features,
    chronological_split,
)
from tests.conftest import requires_data


class TestCalendarFeatures:
    def _frame(self, dates):
        return pd.DataFrame({"auction_date": pd.to_datetime(dates)})

    def test_season_mapping_is_southern_hemisphere(self):
        out = add_calendar_features(self._frame(["2023-01-15", "2023-07-15"]))
        assert out["season"].tolist() == ["summer", "winter"]

    def test_school_holiday_flag_covers_the_easter_break(self):
        out = add_calendar_features(self._frame(["2023-04-10", "2023-05-20"]))
        assert out["school_holiday_flag"].tolist() == [1, 0]

    def test_school_holiday_window_wraps_new_year(self):
        out = add_calendar_features(self._frame(["2023-12-27", "2023-01-05", "2023-03-01"]))
        assert out["school_holiday_flag"].tolist() == [1, 1, 0]


class TestSupplyFeatures:
    def test_counts_auctions_per_day_and_suburb(self):
        df = pd.DataFrame({
            "listing_id": list("abcde"),
            "auction_date": pd.to_datetime(
                ["2023-03-04"] * 3 + ["2023-03-11"] * 2),
            "suburb": ["Richmond", "Richmond", "Kew", "Kew", "Kew"],
        })
        out = add_supply_features(df)
        assert out["auctions_scheduled_that_day"].tolist() == [3, 3, 3, 2, 2]
        assert out["auctions_in_suburb_that_day"].tolist() == [2, 2, 1, 2, 2]


@requires_data
class TestTrailingFeatures:
    def test_clearance_rates_are_valid_probabilities(self, features):
        for col in ("suburb_clearance_l4w", "region_clearance_l4w"):
            vals = features[col].dropna()
            assert vals.between(0, 1).all(), f"{col} outside [0, 1]"

    def test_first_auction_day_has_no_history(self, features):
        """Nothing precedes the first weekend, so the trailing rate is null."""
        first = features["auction_date"].min()
        assert features.loc[features["auction_date"] == first,
                            "suburb_clearance_l4w"].isna().all()

    def test_trailing_features_are_mostly_populated(self, features):
        assert features["suburb_clearance_l4w"].notna().mean() > 0.90
        assert features["region_clearance_l4w"].notna().mean() > 0.99

    def test_guide_vs_median_is_positive_where_defined(self, features):
        vals = features["guide_vs_suburb_median"].dropna()
        assert (vals > 0).all()


@requires_data
class TestChronologicalSplit:
    def test_splits_do_not_overlap_in_time(self, features):
        train, valid, test = chronological_split(features)
        assert train["auction_date"].max() < valid["auction_date"].min()
        assert valid["auction_date"].max() < test["auction_date"].min()

    def test_splits_partition_the_data(self, features):
        train, valid, test = chronological_split(features)
        assert len(train) + len(valid) + len(test) == len(features)
        ids = pd.concat([train, valid, test])["listing_id"]
        assert not ids.duplicated().any()

    def test_every_split_is_non_trivial(self, features):
        train, valid, test = chronological_split(features)
        for name, split in (("train", train), ("valid", valid), ("test", test)):
            assert len(split) > 1000, f"{name} split too small"
            assert split[config.TARGET].nunique() == 2
