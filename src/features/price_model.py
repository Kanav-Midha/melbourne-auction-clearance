"""Expected-value model, used to measure vendor overpricing.

The problem this solves
-----------------------
Whether a vendor's price guide is realistic is the strongest behavioural driver
of whether a property clears. The first attempt at capturing it was

    guide_vs_suburb_median = guide_price_midpoint / suburb_median_price_l90d

which has a standalone AUC of 0.43 -- worse than a coin flip, because it is
confounded. A four-bedroom house guided at $1.6m in a suburb with a $900k median
is not overpriced, it is simply a bigger house. The feature was measuring
"expensive property in a cheap suburb", not "vendor with an expectation
problem".

The fix is to compare the guide against what the property is actually worth:

    price_expectation_gap = log(guide_midpoint) - log(E[value | attributes])

A gradient-boosted tree cannot construct that ratio on its own. Trees split on
thresholds; they cannot divide one input by a learned function of ten others. So
the division has to happen in feature engineering, which is the whole reason
this module exists.

Leakage control
---------------
Two things matter here:

* The value model is fit **only on auctions in the training window**. Scoring a
  2024 property with a model that has seen 2024 sale prices would leak.
* Training rows get **out-of-fold** estimates. If a property's own sale price
  helped fit the model that scores it, the residual shrinks toward zero and the
  feature looks weaker in training than it really is -- which biases every
  downstream decision about whether to keep it.

Known caveat: only *sold* properties have a sale price, so the value model is
fit on a selected sample. Passed-in properties are, by construction, the ones
the market declined at that price. This biases E[value] upward slightly. There
is no clean fix with auction-results data alone; the honest move is to note it
and keep the feature, since the gap is still far better directed than the ratio
it replaced.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src import config

log = logging.getLogger(__name__)

#: Attributes that determine what a property is worth. Deliberately excludes
#: anything about the campaign, the calendar or the macro environment -- this
#: model answers "what is this house worth?", not "will it sell today?".
VALUE_ATTRS = [
    "bedrooms", "bathrooms", "car_spaces", "land_size_sqm", "building_area_sqm",
    "property_age_years", "distance_to_cbd_km", "distance_to_station_km",
    "stations_within_1km", "property_type", "suburb", "council_area",
]
VALUE_CATEGORICALS = ["property_type", "suburb", "council_area"]

VALUE_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": config.RANDOM_SEED,
    "feature_pre_filter": False,
}
VALUE_ROUNDS = 600
N_FOLDS = 5


def _value_frame(df: pd.DataFrame, levels: dict[str, pd.Index] | None = None) -> pd.DataFrame:
    """Build the value-model design matrix with pinned category levels."""
    X = df[VALUE_ATTRS].copy()
    for c in VALUE_ATTRS:
        if c not in VALUE_CATEGORICALS:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    for c in VALUE_CATEGORICALS:
        cats = levels[c] if levels else pd.Index(sorted(df[c].dropna().astype(str).unique()))
        X[c] = pd.Categorical(df[c].astype(str), categories=cats)
    return X


def fit_value_model(train: pd.DataFrame) -> tuple[lgb.Booster, dict[str, pd.Index]]:
    """Fit log-price on attributes, using sold properties in the training window."""
    sold = train[train["sold_price"].notna() & (train["sold_price"] > 0)]
    levels = {
        c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in VALUE_CATEGORICALS
    }
    X = _value_frame(sold, levels)
    y = np.log(sold["sold_price"].to_numpy(dtype=float))

    booster = lgb.train(VALUE_PARAMS, lgb.Dataset(X, y, categorical_feature=VALUE_CATEGORICALS),
                        num_boost_round=VALUE_ROUNDS)
    resid_sd = float(np.std(y - booster.predict(X)))
    log.info("value model fit on %d sold auctions; in-sample residual sd %.3f (log $)",
             len(sold), resid_sd)
    return booster, levels


def _oof_predictions(train: pd.DataFrame, levels: dict[str, pd.Index]) -> pd.Series:
    """Out-of-fold log-value estimates for every training row.

    Folds are random rather than chronological on purpose: the point is to
    remove a row's own influence on the model that scores it, not to simulate
    forecasting. The chronological discipline lives at the split level, where
    the whole value model is confined to the training window.
    """
    sold = train[train["sold_price"].notna() & (train["sold_price"] > 0)]
    X_sold = _value_frame(sold, levels)
    y_sold = np.log(sold["sold_price"].to_numpy(dtype=float))

    oof = pd.Series(np.nan, index=train.index, dtype="float64")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    for fold_train, fold_test in kf.split(X_sold):
        m = lgb.train(
            VALUE_PARAMS,
            lgb.Dataset(X_sold.iloc[fold_train], y_sold[fold_train],
                        categorical_feature=VALUE_CATEGORICALS),
            num_boost_round=VALUE_ROUNDS,
        )
        oof.iloc[
            [train.index.get_loc(i) for i in X_sold.index[fold_test]]
        ] = m.predict(X_sold.iloc[fold_test])

    # Unsold rows never enter the value model, so they get the full-fit estimate.
    return oof


def add_price_expectation(train: pd.DataFrame, others: list[pd.DataFrame]
                          ) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Attach ``expected_log_value`` and ``price_expectation_gap`` to every split.

    Returns the frames in the same order they were supplied.
    """
    booster, levels = fit_value_model(train)

    tr = train.copy()
    oof = _oof_predictions(tr, levels)
    full = pd.Series(booster.predict(_value_frame(tr, levels)), index=tr.index)
    tr["expected_log_value"] = oof.fillna(full)

    outs = []
    for df in others:
        d = df.copy()
        d["expected_log_value"] = booster.predict(_value_frame(d, levels))
        outs.append(d)

    for d in [tr] + outs:
        guide = pd.to_numeric(d["guide_price_midpoint"], errors="coerce")
        d["price_expectation_gap"] = (
            np.log(guide.where(guide > 0)) - d["expected_log_value"]
        )

    log.info("price_expectation_gap: train mean %.4f sd %.4f",
             tr["price_expectation_gap"].mean(), tr["price_expectation_gap"].std())
    return tr, outs


__all__ = [
    "VALUE_ATTRS", "VALUE_CATEGORICALS", "fit_value_model",
    "add_price_expectation",
]
