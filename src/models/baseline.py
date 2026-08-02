"""Logistic-regression baseline.

Kept in the repo on purpose. It is the number every later model has to beat,
and it is what makes the gradient-boosting result meaningful -- "0.74 AUC" means
nothing on its own, "0.74 against a 0.68 linear baseline on the same
chronological split" means something.

    python -m src.models.baseline
"""
from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.models.evaluate import calibration_table, evaluate, save_metrics

log = logging.getLogger(__name__)


def build_pipeline(C: float = 1.0) -> Pipeline:
    """Impute, scale, one-hot, fit.

    ``handle_unknown="infrequent_if_exist"`` matters here: the test split runs a
    year after the training window, and suburbs that never auctioned during
    training do appear later. The notebook version used the default and threw at
    predict time.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=30, sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, C=C, solver="lbfgs")),
    ])


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[config.FEATURES].copy()
    for c in config.CATEGORICAL_FEATURES:
        X[c] = X[c].astype("object")
    return X, df[config.TARGET].to_numpy(dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.data.data_loader import load_auctions, load_stations
    from src.data.weather import build_weather_table
    from src.features.preprocess import build_features
    from src.models.dataset import prepare_splits

    df = build_features(load_auctions(), load_stations(), build_weather_table())
    train, valid, test = prepare_splits(df)

    X_tr, y_tr = _xy(train)
    pipe = build_pipeline(args.C).fit(X_tr, y_tr)

    metrics = {}
    for name, split in (("train", train), ("valid", valid), ("test", test)):
        X, y = _xy(split)
        p = pipe.predict_proba(X)[:, 1]
        metrics[name] = evaluate(y, p, label=f"baseline/{name}")

    X_te, y_te = _xy(test)
    print("\nCalibration on the test split (logistic baseline):")
    print(calibration_table(y_te, pipe.predict_proba(X_te)[:, 1]).to_string(index=False))

    if args.save:
        joblib.dump({"model": pipe, "features": config.FEATURES},
                    config.BASELINE_PATH)
        log.info("saved %s", config.BASELINE_PATH)
        save_metrics(metrics, config.REPORTS_DIR / "metrics_baseline.json")


if __name__ == "__main__":
    main()
