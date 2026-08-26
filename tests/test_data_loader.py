"""Loader and parsing tests.

These cover the messy-input cases that actually occur in the raw extract, each
of which caused a real bug at some point in this project's history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import (
    derive_target,
    load_auctions,
    normalise_suburb,
    parse_currency,
    parse_price_guide,
)
from tests.conftest import requires_data


class TestParseCurrency:
    def test_handles_dollar_signs_and_separators(self):
        s = pd.Series(["$1,250,000", "1250000", "$950,500"])
        out = parse_currency(s)
        assert out.tolist() == [1250000.0, 1250000.0, 950500.0]

    def test_empty_string_becomes_nan_not_zero(self):
        """An unsold property has no price. Zero would be a $0 sale."""
        out = parse_currency(pd.Series(["", "  ", "N/A", "-"]))
        assert out.isna().all()

    def test_returns_float_not_nullable_int(self):
        """Int64 propagates a nullable dtype into every downstream calculation."""
        assert parse_currency(pd.Series(["100", "200"])).dtype == np.dtype("float64")

    def test_garbage_becomes_nan(self):
        assert parse_currency(pd.Series(["POA", "contact agent"])).isna().all()


class TestParsePriceGuide:
    def test_splits_a_range(self):
        out = parse_price_guide(pd.Series(["$900,000 - $990,000"]))
        assert out["guide_price_low"].iloc[0] == 900_000
        assert out["guide_price_high"].iloc[0] == 990_000
        assert out["guide_price_midpoint"].iloc[0] == 945_000

    def test_single_value_guide_has_equal_bounds(self):
        out = parse_price_guide(pd.Series(["$1,100,000"]))
        assert out["guide_price_low"].iloc[0] == 1_100_000
        assert out["guide_price_high"].iloc[0] == 1_100_000
        assert out["guide_price_midpoint"].iloc[0] == 1_100_000

    def test_transposed_range_is_corrected(self):
        out = parse_price_guide(pd.Series(["$990,000 - $900,000"]))
        assert out["guide_price_low"].iloc[0] == 900_000
        assert out["guide_price_high"].iloc[0] == 990_000

    def test_unparseable_guide_is_nan_not_zero(self):
        """A guide of 0 tells the model the property is free."""
        out = parse_price_guide(pd.Series(["Contact agent"]))
        assert out["guide_price_midpoint"].isna().all()


class TestNormaliseSuburb:
    def test_collapses_case_and_whitespace(self):
        out = normalise_suburb(pd.Series([" richmond", "RICHMOND", "Richmond", "richmond  "]))
        assert out.nunique() == 1
        assert out.iloc[0] == "Richmond"

    def test_collapses_internal_whitespace(self):
        out = normalise_suburb(pd.Series(["South  Yarra", "South Yarra"]))
        assert out.nunique() == 1


class TestDeriveTarget:
    def test_sold_codes_map_to_one(self):
        out = derive_target(pd.Series(["S", "SP", "SA"]))
        assert out.tolist() == [1, 1, 1]

    def test_unsold_codes_map_to_zero(self):
        out = derive_target(pd.Series(["PI", "VB"]))
        assert out.tolist() == [0, 0]

    def test_withdrawn_is_null_not_zero(self):
        """Withdrawn lots never went under the hammer, so they are not failures."""
        assert derive_target(pd.Series(["W"])).isna().all()


@requires_data
class TestLoadAuctions:
    def test_no_duplicate_listing_ids(self, auctions):
        assert not auctions["listing_id"].duplicated().any()

    def test_withdrawn_rows_are_dropped(self, auctions):
        assert not auctions["result_code"].eq("W").any()

    def test_suburbs_are_normalised(self, auctions):
        """297 spellings in the raw file collapse to the real suburb count."""
        assert auctions["suburb"].nunique() == 100

    def test_target_is_binary(self, auctions):
        assert set(auctions["sold_at_auction"].unique()) == {0, 1}

    def test_sorted_chronologically(self, auctions):
        assert auctions["auction_date"].is_monotonic_increasing

    def test_clearance_rate_is_plausible(self, auctions):
        """Melbourne clearance sits between 50% and 85% across any real period."""
        assert 0.50 < auctions["sold_at_auction"].mean() < 0.85
