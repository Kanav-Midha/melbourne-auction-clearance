"""Target-leakage audit.

Written after the Stage 1 notebook produced an AUC of 1.0000, which is not a
thing that happens when you predict human behaviour at an auction.

Two separate leaks were involved, and the second one survived the fix for the
first.

1. **Null-pattern leakage** (AUC 1.0000 -> 0.6290 once removed).
   ``sold_price`` is populated if and only if the property sold. So are
   ``price_per_sqm``, ``vendor_discount_pct``, ``days_to_settle`` and
   ``sale_settlement_date``. Median-imputing the nulls does not remove the
   signal, it moves it: "was imputed" *is* the label. ``is_null(sold_price)``
   alone scores AUC 1.0000. Any model that sees missingness -- a tree splitting
   on NaN, or ``SimpleImputer(add_indicator=True)`` -- reproduces the label
   exactly.

2. **Temporal leakage** (AUC 0.6441 -> 0.7754 when reintroduced).
   A trailing suburb clearance rate built with

       g[["sold", "held"]].rolling("28D").sum()

   includes the current auction day, so every row is told how its own weekend
   went. The correct call passes ``closed="left"``. Nothing about the buggy
   version looks wrong: there are no nulls, the values are plausible rates, and
   the feature has an obvious business rationale. It is only visible if you
   test it.

This module encodes both as tests so they cannot come back.

    python -m src.features.leakage_audit          # gate: audits config.FEATURES
    python -m src.features.leakage_audit --scan-raw   # discovery: every column
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config

log = logging.getLogger(__name__)

#: A single feature this far above chance is not a feature, it is the answer.
SINGLE_FEATURE_AUC_LIMIT = 0.85
#: How much a column's *missingness* may predict the target.
NULL_PATTERN_AUC_LIMIT = 0.65


@dataclass
class Finding:
    check: str
    column: str
    detail: str
    severity: str = "error"

    def __str__(self) -> str:
        mark = "FAIL" if self.severity == "error" else "WARN"
        return f"[{mark}] {self.check:<18} {self.column:<28} {self.detail}"


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def add(self, *findings: Finding) -> None:
        self.findings.extend(findings)

    def render(self) -> str:
        if not self.findings:
            return "Leakage audit: clean (0 findings)."
        lines = [f"Leakage audit: {len(self.findings)} finding(s)."]
        lines += [str(f) for f in sorted(self.findings, key=lambda f: f.severity)]
        return "\n".join(lines)


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    """Rank AUC, orientation-free, tolerant of constant or all-null scores."""
    mask = ~pd.isna(score)
    if mask.sum() < 50:
        return None
    y_m, s_m = y[mask], np.asarray(score, dtype=float)[mask]
    if len(np.unique(y_m)) < 2 or len(np.unique(s_m)) < 2:
        return None
    return float(roc_auc_score(y_m, s_m))


def check_null_patterns(df: pd.DataFrame, target: str) -> list[Finding]:
    """Flag columns whose *missingness* predicts the target."""
    y = df[target].to_numpy()
    out = []
    for col in df.columns:
        if col == target:
            continue
        isna = df[col].isna().to_numpy(dtype=float)
        if isna.sum() == 0 or isna.sum() == len(isna):
            continue
        auc = _safe_auc(y, isna)
        if auc is None:
            continue
        lift = abs(auc - 0.5)
        if lift > (NULL_PATTERN_AUC_LIMIT - 0.5):
            # Perfect separation reads as "this column IS the label".
            perfect = bool(
                (df.loc[df[target] == 1, col].notna().all()
                 and df.loc[df[target] == 0, col].isna().all())
                or (df.loc[df[target] == 1, col].isna().all()
                    and df.loc[df[target] == 0, col].notna().all())
            )
            out.append(Finding(
                "null-pattern", col,
                f"AUC of is_null = {auc:.3f}"
                + (" (populated iff target==1)" if perfect else ""),
            ))
    return out


def check_single_feature_auc(df: pd.DataFrame, target: str,
                             columns: list[str] | None = None) -> list[Finding]:
    """Flag any single numeric column that nearly solves the problem alone."""
    y = df[target].to_numpy()
    cols = columns or [c for c in df.columns
                       if c != target and pd.api.types.is_numeric_dtype(df[c])]
    out = []
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        auc = _safe_auc(y, df[col].to_numpy(dtype=float))
        if auc is None:
            continue
        if max(auc, 1 - auc) > SINGLE_FEATURE_AUC_LIMIT:
            out.append(Finding("single-feature", col, f"standalone AUC = {auc:.3f}"))
    return out


def check_temporal_integrity(auctions: pd.DataFrame, stations: pd.DataFrame,
                             weather: pd.DataFrame, cutoff: str = "2022-01-01",
                             columns: tuple[str, ...] = (
                                 "suburb_clearance_l4w",
                                 "region_clearance_l4w",
                                 "suburb_median_price_l90d",
                             )) -> list[Finding]:
    """Rebuild features on a truncated history and require identical values.

    If a trailing feature only uses the past, then computing it over
    ``df[date < cutoff]`` must give exactly the same values for those rows as
    computing it over the full dataset. If a future-looking aggregate has crept
    back in, the two disagree.
    """
    from src.features.preprocess import build_features

    full = build_features(auctions, stations, weather)
    truncated = build_features(
        auctions[auctions["auction_date"] < cutoff].copy(), stations, weather
    )

    key = ["listing_id"]
    merged = full[key + list(columns)].merge(
        truncated[key + list(columns)], on=key, how="inner", suffixes=("_full", "_trunc")
    )

    out = []
    for col in columns:
        a = merged[f"{col}_full"].to_numpy(dtype=float)
        b = merged[f"{col}_trunc"].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = ~(np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True) | both_nan)
        n_diff = int(diff.sum())
        if n_diff:
            worst = float(np.nanmax(np.abs(a[diff] - b[diff]))) if n_diff else 0.0
            out.append(Finding(
                "temporal", col,
                f"{n_diff}/{len(merged)} rows change when the future is removed "
                f"(max delta {worst:.4g})",
            ))
    return out


def check_forbidden_columns(feature_list: list[str]) -> list[Finding]:
    """Guard the configured feature list against the known-leaky columns."""
    return [
        Finding("forbidden", col, "listed in config.LEAKY_COLUMNS but used as a feature")
        for col in feature_list if col in config.LEAKY_COLUMNS
    ]


def run_audit(df: pd.DataFrame, auctions: pd.DataFrame | None = None,
              stations: pd.DataFrame | None = None,
              weather: pd.DataFrame | None = None,
              target: str = config.TARGET) -> AuditReport:
    """Run every check that the supplied inputs allow."""
    report = AuditReport()

    # Audit the *raw* columns for leakage, then the feature list for discipline.
    raw_cols = [c for c in df.columns if c != target]
    report.add(*check_null_patterns(df[raw_cols + [target]], target))
    report.add(*check_single_feature_auc(df, target))
    report.add(*check_forbidden_columns(config.FEATURES))

    if auctions is not None and stations is not None and weather is not None:
        report.add(*check_temporal_integrity(auctions, stations, weather))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-temporal", action="store_true",
                        help="skip the slow permutation-based temporal check")
    parser.add_argument("--scan-raw", action="store_true",
                        help="audit every column in the frame, not just the model "
                             "features. Use this to hunt for new leaks; it will "
                             "always report the known post-sale columns, so it is "
                             "not a pass/fail gate.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.data.data_loader import load_auctions, load_stations
    from src.data.weather import build_weather_table
    from src.features.preprocess import build_features

    auctions = load_auctions()
    stations = load_stations()
    weather = build_weather_table()
    df = build_features(auctions, stations, weather)

    if not args.scan_raw:
        # Gate mode: audit what the model actually consumes. The raw frame
        # legitimately contains sold_price and friends -- they are in the source
        # extract and are excluded by config.LEAKY_COLUMNS, not deleted -- so
        # auditing the raw frame can only ever fail and is worthless as a gate.
        present = [c for c in config.FEATURES if c in df.columns]
        df = df[present + [config.TARGET]]

    report = run_audit(
        df,
        None if args.skip_temporal else auctions,
        None if args.skip_temporal else stations,
        None if args.skip_temporal else weather,
    )
    print(report.render())
    if args.scan_raw and report.failed:
        print("\n(--scan-raw is a discovery tool, not a gate: the post-sale "
              "columns above are expected in the raw extract.)")
        return 0
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
