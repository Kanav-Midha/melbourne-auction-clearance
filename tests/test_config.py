"""Configuration must import on a filesystem it cannot write to.

The serving container ships only ``models/`` and ``src/`` and runs as a non-root
user. An unconditional ``mkdir`` at import time raised PermissionError there, so
uvicorn never bound a port and the container exited before answering a single
request -- with no error visible except a health check that timed out.
"""
from __future__ import annotations

import importlib
import pathlib

import pytest

from src import config


def test_import_survives_an_unwritable_filesystem(monkeypatch):
    """Re-import config with every mkdir denied; it must not raise."""
    def denied(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(pathlib.Path, "mkdir", denied)
    reloaded = importlib.reload(config)
    assert reloaded.MODEL_PATH.name == "clearance_lgbm.joblib"


@pytest.fixture(autouse=True)
def _restore_config():
    """Reload config unpatched afterwards, so later tests see real directories."""
    yield
    importlib.reload(config)


def test_ensure_working_dirs_is_idempotent():
    config._ensure_working_dirs()
    config._ensure_working_dirs()
    assert config.MODEL_DIR.exists()


def test_paths_are_anchored_to_the_repo_root():
    """Paths must not depend on the process working directory.

    The API, the notebooks and the CLI all import this module from different
    directories; a relative path here breaks two of the three.
    """
    assert config.PROJECT_ROOT.is_absolute()
    assert (config.PROJECT_ROOT / "src" / "config.py").exists()
    for p in (config.RAW_DIR, config.MODEL_DIR, config.REPORTS_DIR):
        assert p.is_absolute()
        assert config.PROJECT_ROOT in p.parents


def test_no_leaky_column_is_listed_as_a_feature():
    assert not set(config.FEATURES) & set(config.LEAKY_COLUMNS)
