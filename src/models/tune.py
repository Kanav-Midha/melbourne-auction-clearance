"""Optuna hyperparameter search for the LightGBM clearance model.

The search optimises PR-AUC on the chronological validation split, with early
stopping inside each trial. Two deliberate choices:

* **No cross-validation.** K-fold on a time series leaks the future into the
  past. The validation split is the six months immediately after training,
  which is the closest analogue to how the model is actually used.
* **PR-AUC, not ROC-AUC.** The base rate is ~0.70, so ROC-AUC is dominated by
  the easy negatives; PR-AUC tracks the decisions an agent cares about.

    python -m src.models.tune --trials 60
"""
from __future__ import annotations

import argparse
import json
import logging

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import average_precision_score

from src import config
from src.models.dataset import load_splits

log = logging.getLogger(__name__)

PARAMS_PATH = config.MODEL_DIR / "best_params.json"

FIXED = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_jobs": -1,
    "seed": config.RANDOM_SEED,
}


def suggest(trial: optuna.Trial) -> dict:
    """The search space.

    ``num_leaves`` and ``min_child_samples`` are the two that actually move the
    needle here; the rest mostly trade a little variance. ``max_cat_to_onehot``
    is pinned low so LightGBM uses its sorted-category split for ``suburb``
    rather than one-hot, which is what makes the 100-level column usable.
    """
    return {
        **FIXED,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 192, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 5.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "cat_smooth": trial.suggest_int("cat_smooth", 5, 100),
        "max_cat_to_onehot": 4,
    }


def make_objective(X_tr, y_tr, X_va, y_va, max_rounds: int):
    dtrain = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, y_va, reference=dtrain, free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        booster = lgb.train(
            params, dtrain, num_boost_round=max_rounds,
            valid_sets=[dvalid], valid_names=["valid"],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        p = booster.predict(X_va, num_iteration=booster.best_iteration)
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        return float(average_precision_score(y_va, p))

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--max-rounds", type=int, default=3000)
    parser.add_argument("--timeout", type=int, default=None, help="seconds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    (X_tr, X_va, _), (y_tr, y_va, _), _ = load_splits()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED, n_startup_trials=12),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=12),
        study_name="clearance_lgbm",
    )
    study.optimize(
        make_objective(X_tr, y_tr, X_va, y_va, args.max_rounds),
        n_trials=args.trials, timeout=args.timeout, show_progress_bar=False,
    )

    best = {**FIXED, **study.best_params, "max_cat_to_onehot": 4}
    payload = {
        "params": best,
        "best_iteration": study.best_trial.user_attrs.get("best_iteration"),
        "valid_pr_auc": round(study.best_value, 5),
        "n_trials": len(study.trials),
    }
    PARAMS_PATH.write_text(json.dumps(payload, indent=2))

    log.info("best valid PR-AUC %.4f after %d trials", study.best_value, len(study.trials))
    log.info("best iteration: %s", payload["best_iteration"])
    log.info("wrote %s", PARAMS_PATH)

    top = sorted(study.trials, key=lambda t: (t.value or 0), reverse=True)[:5]
    log.info("top 5 trials: %s", [round(t.value, 4) for t in top if t.value])


if __name__ == "__main__":
    main()
