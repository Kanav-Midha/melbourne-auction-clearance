"""Regression tests for the two leaks that inflated this project's early numbers.

Both of these were real. Notebook 01 reported AUC 1.0000 and notebook 02
reported 0.7754; the honest number is around 0.65. These tests exist so that
neither can return silently.

The important test here is ``test_temporal_check_detects_a_reintroduced_leak``.
A check that never fails proves nothing, so the buggy implementation is
monkeypatched back in and the audit is required to catch it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import preprocess
from src.features.leakage_audit import (
    check_forbidden_columns,
    check_null_patterns,
    check_temporal_integrity,
    run_audit,
)
from tests.conftest import requires_data


class TestForbiddenColumns:
    def test_feature_list_contains_no_leaky_columns(self):
        """The guard that stops sold_price being re-added by accident."""
        assert check_forbidden_columns(config.FEATURES) == []

    def test_guard_fires_when_a_leaky_column_is_added(self):
        findings = check_forbidden_columns(config.FEATURES + ["sold_price"])
        assert len(findings) == 1
        assert findings[0].column == "sold_price"


class TestNullPatterns:
    def test_detects_a_column_populated_only_for_positives(self):
        df = pd.DataFrame({
            "target": [1, 1, 0, 0] * 50,
            "post_sale_value": [100.0, 120.0, np.nan, np.nan] * 50,
        })
        findings = check_null_patterns(df, "target")
        assert [f.column for f in findings] == ["post_sale_value"]
        assert "populated iff target==1" in findings[0].detail

    def test_ignores_a_column_missing_at_random(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 2000)
        col = rng.normal(size=2000)
        col[rng.random(2000) < 0.3] = np.nan
        df = pd.DataFrame({"target": y, "harmless": col})
        assert check_null_patterns(df, "target") == []


@requires_data
class TestShippedPipeline:
    def test_trailing_window_excludes_the_current_day(self, features):
        """The bug was a missing closed='left'. Verify the window is causal.

        For one suburb-day, the feature must equal the clearance rate over the
        preceding 28 days *excluding* that day.
        """
        df = features
        suburb = "Richmond"
        rows = df[(df["suburb"] == suburb) & df["suburb_clearance_l4w"].notna()]
        probe_date = rows["auction_date"].iloc[len(rows) // 2]

        window = df[
            (df["suburb"] == suburb)
            & (df["auction_date"] < probe_date)
            & (df["auction_date"] >= probe_date - pd.Timedelta("28D"))
        ]
        expected = window[config.TARGET].mean()
        actual = rows[rows["auction_date"] == probe_date]["suburb_clearance_l4w"].iloc[0]
        assert actual == pytest.approx(expected)

    def test_temporal_integrity_passes_on_the_shipped_pipeline(
        self, auctions, stations, weather
    ):
        assert check_temporal_integrity(auctions, stations, weather) == []

    def test_temporal_check_detects_a_reintroduced_leak(
        self, auctions, stations, weather, monkeypatch
    ):
        """Put the original bug back and require the audit to catch it.

        Without this, the check above could pass because the check is broken
        rather than because the pipeline is correct.
        """
        original = preprocess.add_trailing_features

        def buggy(df: pd.DataFrame) -> pd.DataFrame:
            out = original(df)
            daily = (
                df.groupby(["suburb", "auction_date"])[config.TARGET]
                .agg(sold="sum", held="size")
                .reset_index()
                .sort_values(["suburb", "auction_date"])
            )
            pieces = []
            for name, grp in daily.groupby("suburb", sort=False):
                g = grp.set_index("auction_date")
                roll = g[["sold", "held"]].rolling("28D").sum()  # no closed="left"
                pieces.append(
                    (roll["sold"] / roll["held"]).rename("rate")
                    .reset_index().assign(suburb=name)
                )
            out["suburb_clearance_l4w"] = out[["suburb", "auction_date"]].merge(
                pd.concat(pieces, ignore_index=True),
                on=["suburb", "auction_date"], how="left",
            )["rate"].to_numpy()
            return out

        monkeypatch.setattr(preprocess, "add_trailing_features", buggy)
        findings = check_temporal_integrity(auctions, stations, weather)

        assert findings, "the temporal check failed to detect a known leak"
        assert any(f.column == "suburb_clearance_l4w" for f in findings)

    def test_price_model_does_not_see_the_test_years(self, features):
        """The value model must be fit on training rows only.

        Tested without assuming which way the market moved: a model that has
        seen the test period fits it noticeably better than one that has not.
        An earlier version of this test asserted that the model under-predicts
        2024 prices, which failed -- Melbourne prices peaked in 2022 and fell
        into 2024, so a train-only model over-predicts. That assertion was
        measuring the market, not the leak.
        """
        import numpy as np

        from src.features.preprocess import chronological_split
        from src.features.price_model import _value_frame, fit_value_model

        train, _valid, test = chronological_split(features)
        assert train["auction_date"].max() <= pd.Timestamp(config.TRAIN_END)

        sold_test = test[test["sold_price"].notna()]
        actual = np.log(sold_test["sold_price"].to_numpy(dtype=float))

        honest, levels = fit_value_model(train)
        honest_resid = np.std(actual - honest.predict(_value_frame(sold_test, levels)))

        # A model that HAS seen the test period should fit it strictly better.
        cheating, cheat_levels = fit_value_model(pd.concat([train, test]))
        cheat_resid = np.std(actual - cheating.predict(_value_frame(sold_test, cheat_levels)))

        assert cheat_resid < honest_resid, (
            f"fitting on the test period did not improve the fit "
            f"({cheat_resid:.4f} vs {honest_resid:.4f}); the honest model may "
            f"already be seeing test data"
        )

    def test_no_feature_solves_the_problem_alone(self, features):
        """No single feature should reach AUC 0.85. sold_price used to hit 1.0."""
        from sklearn.metrics import roc_auc_score

        y = features[config.TARGET].to_numpy()
        offenders = []
        for col in config.NUMERIC_FEATURES:
            if col not in features.columns:
                continue
            s = pd.to_numeric(features[col], errors="coerce").to_numpy(dtype=float)
            mask = ~np.isnan(s)
            if mask.sum() < 100 or len(np.unique(s[mask])) < 2:
                continue
            auc = roc_auc_score(y[mask], s[mask])
            if max(auc, 1 - auc) > 0.85:
                offenders.append((col, round(auc, 4)))
        assert offenders == []

    def test_full_audit_is_clean_on_the_feature_set(self, features):
        """Audit the row-wise features. The split-aware ones are added later by
        models.dataset.prepare_splits and are covered by test_price_model."""
        cols = [c for c in config.FEATURES if c not in config.SPLIT_AWARE_FEATURES]
        report = run_audit(features[cols + [config.TARGET]])
        assert not report.failed, report.render()
