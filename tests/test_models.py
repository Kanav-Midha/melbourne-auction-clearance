"""Model-artefact consistency tests.

These guard the seams between the three ways a model gets built -- a fresh
clone, a tuning run, and the Docker image -- which is where configuration
silently drifts apart.

The bug that motivated this file: ``train.DEFAULT_PARAMS`` still held an earlier
tuning round's values (``num_leaves: 151``) long after the search space had been
narrowed to a maximum of 64. A normal checkout never noticed, because
``models/best_params.json`` is committed and takes precedence. The Docker build
copied only ``src/``, so it fell through to the stale defaults and would have
shipped a different model than the README documents.
"""
from __future__ import annotations

import json

import optuna
import pytest

from src import config
from src.models import train, tune
from tests.conftest import requires_model


@pytest.fixture(scope="module")
def best_params() -> dict:
    if not tune.PARAMS_PATH.exists():
        pytest.skip("no tuning run on disk; run `make tune`")
    return json.loads(tune.PARAMS_PATH.read_text())


class TestDefaultsMatchTunedParams:
    """``DEFAULT_PARAMS`` is the Docker build's fallback; it must not go stale."""

    def test_every_tuned_value_is_mirrored(self, best_params):
        tuned = best_params["params"]
        drift = {
            k: (v, train.DEFAULT_PARAMS.get(k))
            for k, v in tuned.items()
            if k not in train.DEFAULT_PARAMS
            or not _close(train.DEFAULT_PARAMS[k], v)
        }
        assert not drift, (
            "train.DEFAULT_PARAMS has drifted from models/best_params.json "
            f"(key: (tuned, default)): {drift}"
        )

    def test_round_count_is_mirrored(self, best_params):
        assert int(best_params["best_iteration"]) == train.DEFAULT_ROUNDS


def _close(a, b, tol: float = 1e-5) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    return a == b


class TestParamsLieInsideTheSearchSpace:
    """Checked-in parameters must be reachable by the search that produced them.

    If the space is narrowed but the params file is not regenerated, the repo
    documents a model the tuner can no longer find -- which is exactly how the
    stale ``num_leaves: 151`` survived.
    """

    def test_tuned_params_are_reachable(self, best_params):
        space = _search_space_bounds()
        tuned = best_params["params"]
        out_of_range = {
            name: (tuned[name], bounds)
            for name, bounds in space.items()
            if name in tuned and not (bounds[0] <= tuned[name] <= bounds[1])
        }
        assert not out_of_range, (
            "checked-in parameters fall outside tune.suggest()'s current space; "
            f"re-run `make tune`: {out_of_range}"
        )

    def test_defaults_are_reachable(self):
        space = _search_space_bounds()
        out_of_range = {
            name: (train.DEFAULT_PARAMS[name], bounds)
            for name, bounds in space.items()
            if name in train.DEFAULT_PARAMS
            and not (bounds[0] <= train.DEFAULT_PARAMS[name] <= bounds[1])
        }
        assert not out_of_range, out_of_range


def _search_space_bounds() -> dict[str, tuple[float, float]]:
    """Recover each tunable's range by introspecting ``tune.suggest``."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    tune.suggest(trial)
    bounds = {}
    for name, dist in trial.distributions.items():
        low, high = getattr(dist, "low", None), getattr(dist, "high", None)
        if low is not None and high is not None:
            bounds[name] = (low, high)
    return bounds


@requires_model
class TestSavedBundle:
    """The API unpacks this bundle by key; a rename here is a 500 in production."""

    @pytest.fixture(scope="class")
    def bundle(self):
        import joblib
        return joblib.load(config.MODEL_PATH)

    def test_has_every_key_the_api_reads(self, bundle):
        required = {
            "booster", "features", "categorical_features", "category_levels",
            "value_booster", "value_levels", "n_rounds", "trained_through",
            "train_rows", "test_metrics", "feature_importance",
        }
        assert required <= set(bundle), f"missing: {required - set(bundle)}"

    def test_feature_list_matches_the_booster(self, bundle):
        assert list(bundle["booster"].feature_name()) == list(bundle["features"])

    def test_feature_list_matches_config(self, bundle):
        assert list(bundle["features"]) == list(config.FEATURES)

    def test_category_levels_cover_every_categorical(self, bundle):
        assert set(bundle["category_levels"]) == set(config.CATEGORICAL_FEATURES)

    def test_no_leaky_column_reached_the_model(self, bundle):
        leaked = set(bundle["features"]) & set(config.LEAKY_COLUMNS)
        assert not leaked, f"leaky columns in the shipped model: {leaked}"

    def test_reported_metrics_are_plausible(self, bundle):
        m = bundle["test_metrics"]
        assert 0.55 < m["roc_auc"] < 0.85, "suspicious held-out AUC"
        assert m["ece"] < 0.06, "shipped model is poorly calibrated"
