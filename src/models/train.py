"""Train the production LightGBM clearance model.

Two-stage fit, which is the part worth explaining in a review:

1. Fit on ``train`` with early stopping against ``valid`` to find the round
   count. This is the only honest way to pick ``n_estimators`` on a time series.
2. Refit on ``train + valid`` at that round count (scaled for the extra data)
   so the shipped model has seen the most recent six months. A model that stops
   learning at the training cut-off is a model that has never seen the current
   rate environment.

``test`` is touched exactly once, at the end. It is the 2024 calendar year --
a full year after the training window closes, which is the real deployment gap.

    python -m src.models.train --save
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.models.dataset import load_splits
from src.models.evaluate import (
    calibration_table,
    evaluate,
    evaluate_by_segment,
    save_metrics,
)
from src.models.tune import PARAMS_PATH

log = logging.getLogger(__name__)

#: Used when models/best_params.json is absent, so `make train` works on a
#: fresh clone without a 20-minute search first. These ARE the tuned values --
#: kept in sync with models/best_params.json by tests/test_models.py, because
#: a stale copy here silently trains a different model inside the Docker image
#: (which builds from src/ plus the params file, not from a full checkout).
DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_jobs": -1,
    "seed": config.RANDOM_SEED,
    "feature_pre_filter": False,
    "learning_rate": 0.020792,
    "num_leaves": 44,
    "max_depth": 8,
    "min_child_samples": 132,
    "feature_fraction": 0.840302,
    "bagging_fraction": 0.681145,
    "bagging_freq": 4,
    "lambda_l1": 1.013598,
    "lambda_l2": 0.001381,
    "min_split_gain": 0.264147,
    "cat_smooth": 90,
    "max_cat_to_onehot": 4,
}
DEFAULT_ROUNDS = 92


def load_params() -> tuple[dict, int]:
    """Prefer tuned parameters if a search has been run."""
    if PARAMS_PATH.exists():
        blob = json.loads(PARAMS_PATH.read_text())
        log.info("using tuned params from %s (valid PR-AUC %.4f)",
                 PARAMS_PATH.name, blob.get("valid_pr_auc", float("nan")))
        return blob["params"], int(blob.get("best_iteration") or DEFAULT_ROUNDS)
    log.info("no tuned params found; using checked-in defaults")
    return dict(DEFAULT_PARAMS), DEFAULT_ROUNDS


def feature_importance(booster: lgb.Booster) -> pd.DataFrame:
    """Gain-based importance. Split counts flatter high-cardinality columns."""
    return (
        pd.DataFrame({
            "feature": booster.feature_name(),
            "gain": booster.feature_importance("gain"),
            "split": booster.feature_importance("split"),
        })
        .sort_values("gain", ascending=False)
        .assign(gain_pct=lambda d: (100 * d["gain"] / d["gain"].sum()).round(2))
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-refit", action="store_true",
                        help="skip the train+valid refit (for ablation)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    (X_tr, X_va, X_te), (y_tr, y_va, y_te), (tr_df, va_df, te_df) = load_splits()
    params, fallback_rounds = load_params()

    # --- stage 1: find the round count ---------------------------------------
    dtrain = lgb.Dataset(X_tr, y_tr)
    dvalid = lgb.Dataset(X_va, y_va, reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=3000,
        valid_sets=[dtrain, dvalid], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    best_rounds = booster.best_iteration or fallback_rounds
    log.info("early stopping selected %d rounds", best_rounds)

    metrics = {
        "baseline_reference": "see reports/metrics_baseline.json",
        "train": evaluate(y_tr, booster.predict(X_tr), "lgbm/train"),
        "valid": evaluate(y_va, booster.predict(X_va), "lgbm/valid"),
    }

    # --- stage 2: refit on train + valid -------------------------------------
    if args.no_refit:
        final = booster
        final_rounds = best_rounds
    else:
        X_full = pd.concat([X_tr, X_va], axis=0)
        y_full = np.concatenate([y_tr, y_va])
        # More data supports slightly more trees; scale by the row ratio.
        final_rounds = int(round(best_rounds * len(X_full) / len(X_tr)))
        final = lgb.train(params, lgb.Dataset(X_full, y_full),
                          num_boost_round=final_rounds,
                          callbacks=[lgb.log_evaluation(0)])
        log.info("refit on train+valid (%d rows, %d rounds)", len(X_full), final_rounds)

    # --- the one look at the test split --------------------------------------
    p_te = final.predict(X_te)
    metrics["test"] = evaluate(y_te, p_te, "lgbm/test")

    imp = feature_importance(final)
    print("\nTop 15 features by gain:")
    print(imp.head(15).to_string(index=False))

    print("\nCalibration on the test split (2024):")
    print(calibration_table(y_te, p_te).to_string(index=False))

    seg = te_df.assign(_y=y_te, _p=p_te)
    print("\nTest performance by region:")
    print(evaluate_by_segment(seg, "_y", "_p", "region_name").to_string(index=False))

    if args.save:
        # The API has to rebuild price_expectation_gap for a single property, so
        # the value model travels with the classifier. Shipping them separately
        # is how online and offline features drift apart.
        from src.features.price_model import fit_value_model
        value_booster, value_levels = fit_value_model(tr_df)

        bundle = {
            "booster": final,
            "value_booster": value_booster,
            "value_levels": value_levels,
            "features": config.FEATURES,
            "numeric_features": config.NUMERIC_FEATURES,
            "categorical_features": config.CATEGORICAL_FEATURES,
            # The API must encode a single record with exactly these levels.
            "category_levels": {
                c: list(X_tr[c].cat.categories) for c in config.CATEGORICAL_FEATURES
            },
            "params": params,
            "n_rounds": final_rounds,
            "trained_through": str(config.VALID_END),
            "train_rows": int(len(X_tr) + (0 if args.no_refit else len(X_va))),
            "test_metrics": metrics["test"].to_dict(),
            "feature_importance": imp.to_dict("records"),
        }
        joblib.dump(bundle, config.MODEL_PATH)
        log.info("saved %s", config.MODEL_PATH)

        from src.api.context import build_context_store, save_context
        from src.data.data_loader import load_auctions as _la
        from src.data.data_loader import load_stations as _ls
        from src.data.weather import build_weather_table as _bw
        from src.features.preprocess import build_features as _bf
        save_context(build_context_store(_bf(_la(), _ls(), _bw())))

        save_metrics(metrics, config.METRICS_PATH)
        imp.to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)
        calibration_table(y_te, p_te).to_csv(
            config.REPORTS_DIR / "calibration_test.csv", index=False)


if __name__ == "__main__":
    main()
