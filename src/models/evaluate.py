"""Model evaluation.

Discrimination is not the point of this model. An agent asking "will this
clear?" needs a *calibrated* probability -- if the model says 70%, roughly 70%
of those properties should sell -- because the number feeds a vendor
conversation, not a ranking. So Brier score and calibration error are reported
alongside AUC, and the headline metric for model selection is PR-AUC against a
base rate near 0.70.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

log = logging.getLogger(__name__)


@dataclass
class Metrics:
    n: int
    base_rate: float
    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float
    ece: float
    lift_top_decile: float

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}

    def __str__(self) -> str:
        return (
            f"n={self.n:>6}  base={self.base_rate:.3f}  "
            f"ROC-AUC={self.roc_auc:.4f}  PR-AUC={self.pr_auc:.4f}  "
            f"Brier={self.brier:.4f}  ECE={self.ece:.4f}  "
            f"lift@10%={self.lift_top_decile:.2f}x"
        )


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Mean |predicted - observed| across equal-frequency probability bins.

    Equal-frequency rather than equal-width: predictions cluster around the base
    rate, so equal-width bins leave the tails almost empty and the number gets
    dominated by noise.
    """
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    total = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        total += len(b) * abs(p[b].mean() - y[b].mean())
    return float(total / len(p))


def lift_at_decile(y: np.ndarray, p: np.ndarray, q: float = 0.9) -> float:
    """Clearance rate among the top-scoring decile, relative to the base rate."""
    cut = np.quantile(p, q)
    top = p >= cut
    if top.sum() == 0 or y.mean() == 0:
        return float("nan")
    return float(y[top].mean() / y.mean())


def evaluate(y_true, y_prob, label: str = "") -> Metrics:
    """Compute the full metric set for one split."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)

    m = Metrics(
        n=len(y),
        base_rate=float(y.mean()),
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p)),
        ece=expected_calibration_error(y, p),
        lift_top_decile=lift_at_decile(y, p),
    )
    if label:
        log.info("%-22s %s", label, m)
    return m


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed clearance by probability decile.

    This is the table to put in front of a stakeholder: it answers "when the
    model says 80%, what actually happens?" without mentioning AUC.
    """
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=int),
                       "p": np.asarray(y_prob, dtype=float)})
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    out = (
        df.groupby("bin")
        .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"),
             p_min=("p", "min"), p_max=("p", "max"))
        .reset_index(drop=True)
    )
    out["gap"] = out["observed"] - out["predicted"]
    return out.round(4)


def evaluate_by_segment(df: pd.DataFrame, y_col: str, p_col: str,
                        segment: str) -> pd.DataFrame:
    """Per-segment metrics -- where a headline average hides a broken slice."""
    rows = []
    for name, grp in df.groupby(segment, observed=True):
        if len(grp) < 100 or grp[y_col].nunique() < 2:
            continue
        y, p = grp[y_col].to_numpy(dtype=int), grp[p_col].to_numpy(dtype=float)
        rows.append({
            segment: name,
            "n": len(grp),
            "base_rate": round(float(y.mean()), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "mean_pred": round(float(p.mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def save_metrics(metrics: dict, path) -> None:
    """Write the metrics blob that the README table is generated from."""
    payload = {
        k: (v.to_dict() if isinstance(v, Metrics) else v) for k, v in metrics.items()
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log.info("wrote %s", path)


__all__ = [
    "Metrics", "evaluate", "calibration_table", "evaluate_by_segment",
    "expected_calibration_error", "lift_at_decile", "save_metrics",
]
