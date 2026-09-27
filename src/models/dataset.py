"""Shared dataset assembly for the model scripts.

Both ``tune.py`` and ``train.py`` need the same frames prepared the same way.
Before this existed they each built their own, and they disagreed about whether
``season`` was categorical -- which is why the tuned parameters did not
reproduce the first time they were used.
"""
from __future__ import annotations

import logging

import pandas as pd

from src import config

log = logging.getLogger(__name__)


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select the feature columns and coerce categoricals to pandas ``category``.

    LightGBM consumes ``category`` dtype natively, which avoids one-hot blowing
    up ``suburb`` into 100 sparse columns. The categories are pinned to the
    union across all splits so train and test share an identical encoding --
    LightGBM keys on the integer codes, and unpinned categories silently remap
    them between frames.
    """
    X = df[config.FEATURES].copy()
    for c in config.NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X


def pin_categories(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Give every frame the same category set for each categorical column."""
    out = [f.copy() for f in frames]
    for col in config.CATEGORICAL_FEATURES:
        levels = pd.Index(
            sorted(set().union(*[set(f[col].dropna().astype(str).unique()) for f in out]))
        )
        dtype = pd.CategoricalDtype(categories=levels, ordered=False)
        for f in out:
            f[col] = f[col].astype(str).where(f[col].notna()).astype(dtype)
    return out


def prepare_splits(df: pd.DataFrame):
    """Split chronologically, then attach the split-aware price features.

    ``price_expectation_gap`` cannot be built inside ``build_features`` because
    it needs to know which rows are training rows -- the value model is fit on
    the training window only. Doing it here keeps that boundary explicit instead
    of burying a train/test distinction inside a "pure" feature function.
    """
    from src.features.preprocess import chronological_split
    from src.features.price_model import add_price_expectation

    train, valid, test = chronological_split(df)
    train, (valid, test) = add_price_expectation(train, [valid, test])
    return train, valid, test


def load_splits(verbose: bool = True):
    """Build features and return ``(train, valid, test)`` model frames + labels."""
    from src.data.data_loader import load_auctions, load_stations
    from src.data.weather import build_weather_table
    from src.features.preprocess import build_features

    df = build_features(load_auctions(), load_stations(), build_weather_table())
    train, valid, test = prepare_splits(df)

    Xs = pin_categories([to_model_frame(s) for s in (train, valid, test)])
    ys = [s[config.TARGET].to_numpy(dtype=int) for s in (train, valid, test)]

    if verbose:
        log.info("features: %d numeric, %d categorical",
                 len(config.NUMERIC_FEATURES), len(config.CATEGORICAL_FEATURES))
    return Xs, ys, (train, valid, test)


__all__ = ["to_model_frame", "pin_categories", "prepare_splits", "load_splits"]
